"""Failure audit through actual adapters with caller-supplied, deterministic fake engines."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from ricepo.audit import GenerationTruncated, attempt, fingerprint, stable_seed
from ricepo.backends import VLLMGenerator
from ricepo.formats import NativeTagParser
from ricepo.hf_runtime import HFNativeGenerator
from ricepo.rollout import BranchRequest, MalformedGenerationError, RolloutRules
from test_formats import PieceTokenizer
from test_rollout import anchor_for, make_runner


def _events(root, kind):
    records = (json.loads(p.read_text("utf-8")) for p in root.glob("events/*.json"))
    return [r for r in records if r["kind"] == kind]


PREFIX, SEED, VERSION = (210, 214, 214), 43, "frozen-asset/policy-1"


def _adapter(kind, ids, parser, *, length=False):
    tokenizer = PieceTokenizer()
    tokenizer.bos_token_id = None
    if kind == "vllm":
        class Engine:
            def generate(self, *, prompts, sampling_params):
                assert prompts == [{"prompt_token_ids": list(PREFIX)}]
                assert sampling_params == {"n": 1, "seed": SEED}
                completion = SimpleNamespace(token_ids=list(ids), finish_reason="length" if length else "stop")
                return [SimpleNamespace(prompt_token_ids=list(PREFIX), outputs=[completion])]

        return VLLMGenerator(engine=Engine(), policy_version_getter=lambda: VERSION,
                             sampling_params_factory=dict, sampling_settings={"n": 1},
                             generation_parser=parser, engine_generate_kwargs={})
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Model:
        training = False
        def generate(self, *, input_ids, use_model_defaults, **kwargs):
            assert tuple(input_ids[0].tolist()) == PREFIX
            assert use_model_defaults is False
            assert torch.initial_seed() == SEED
            return torch.tensor([[*PREFIX, *ids]])

    generator = HFNativeGenerator(model=Model(), tokenizer=tokenizer, device="cpu",
                                  policy_version_getter=lambda: VERSION,
                                  max_new_tokens=len(ids) if length else len(ids) + 5,
                                  max_context_tokens=64, prefilled_think=False, precision_context=nullcontext)
    generator.parser = parser
    return generator


@pytest.mark.parametrize("kind", ["hf", "vllm"])
@pytest.mark.parametrize(("ids", "category"), [
    ((1, 2, 3, 4, 6, 7), "empty_span"),
    ((9, 3, 4, 5, 6, 7), "tag_inside_token"),
    ((1, 8, 2, 3, 4, 5, 6, 7), "ambiguous_tokenizer_boundary"),
    ((2, 3, 4, 5, 6, 7), "tag_structure"),
    ((2, 2), "generation_length"),
])
def test_returned_ids_are_durable_before_parse_and_survive_failure(tmp_path, kind, ids, category):
    request_id = fingerprint([VERSION, "q", "t", 1, SEED, None, None])
    native_parser = NativeTagParser(PieceTokenizer(), prefilled_think=False, malformed_policy="raise")
    parsed = []
    def parser(tokens, reason):
        # Observe durability at parser entry, not just after failure cleanup.
        (raw,) = _events(root, "raw_completion")
        assert raw["token_ids"] == list(ids)
        parsed.append((tokens, reason))
        return native_parser(tokens, reason)

    length = category == "generation_length"
    generator = _adapter(kind, ids, parser, length=length)
    runner = make_runner(generator=generator, rules=RolloutRules(frozenset({"stop"}), 20, 64, depth=1))
    with pytest.raises(GenerationTruncated if length else MalformedGenerationError):
        with attempt(tmp_path) as root:
            runner.run(query_id="q", trajectory_id="t", original_query="original",
                       initial_prefix_token_ids=PREFIX, seeds=(SEED,), old_policy_version=VERSION)
    assert bool(parsed) is (not length)
    (raw,) = _events(root, "raw_completion")
    (failed,) = _events(root, "failed")
    assert raw["phase"] == failed["phase"] == "original_generation"
    assert raw["request_id"] == failed["request_id"] == request_id
    assert raw["operation_id"] == failed["operation_id"]
    assert raw["prefix_sha256"] == fingerprint(PREFIX)
    assert raw["prefix_token_ids"] == list(PREFIX)
    assert (raw["seed"], raw["depth"], raw["policy_version"]) == (SEED, 1, VERSION)
    assert (raw["query_id"], raw["trajectory_id"], raw["token_ids"]) == ("q", "t", list(ids))
    assert raw["input_tokens"] == len(PREFIX) and raw["generated_tokens"] == len(ids)
    expected_reason = ("length" if length else "stop") if kind == "vllm" else ("length_or_stop" if length else "eos")
    assert raw["finish_reason"] == expected_reason
    assert failed["category"] == category
    assert json.loads((root / "failure.json").read_text("utf-8"))["category"] == category
    assert runner.retriever.calls == [] and runner.reward_scorer.calls == []


def test_request_order_and_branch_count_preserve_original_streams_and_histories():
    root_seed, iteration, domain = 13, 2, "fixture"
    def schedule(order, branch_count):
        runner = make_runner()
        # The reused SeedGenerator includes repeated native IDs and the seed
        # itself in its summary. Only this scorer needs adapting to 63-bit seeds.
        runner.reward_scorer = SimpleNamespace(score=lambda q, docs: 0.25)
        originals, alternatives = {}, {}
        for q in order:
            seeds = tuple(stable_seed(root_seed, "original", iteration, domain, q, 0, d) for d in range(1, 6))
            trajectory = runner.run(query_id=q, trajectory_id=f"{q}-0", original_query=q,
                                    initial_prefix_token_ids=(41, 42), seeds=seeds,
                                    old_policy_version=runner.generator.policy_version)
            originals[q] = (seeds, trajectory)
            anchor = anchor_for(trajectory, 3)
            requests = [BranchRequest(f"{q}-0-d3-b{b}", tuple(
                stable_seed(root_seed, "branch", trajectory.policy_version, domain, q,
                            trajectory.trajectory_id, anchor.depth, b, d) for d in range(3, 6)))
                for b in range(1, branch_count + 1)]
            traces = runner.branch(original_query=q, original=trajectory, anchor=anchor, requests=requests).trajectories
            alternatives.update({trace.trajectory_id: trace for trace in traces})
        return originals, alternatives

    original_a, branches_a = schedule(("q1", "q2"), 1)
    original_b, branches_b = schedule(("q2", "q1"), 3)
    assert original_a == original_b
    assert all(trace == branches_b[identity] for identity, trace in branches_a.items())
    assert set(branches_a) < set(branches_b)

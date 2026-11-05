"""Backend contract tests using supplied fakes; no model downloads or engine startup."""

from math import isnan
from types import SimpleNamespace

import pytest

from ricepo.backends import BackendContractError, TransformersEntropyProvider, VLLMGenerator
from ricepo.contracts import Generation, Span, Trajectory
from ricepo.rollout import MalformedGenerationError, PolicyVersionError


def parser(ids, reason):
    return Generation(ids, Span(0, 1), Span(1, len(ids)), "explicit parser summary", reason)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params, kwargs))
        return [SimpleNamespace(
            prompt_token_ids=prompts[0]["prompt_token_ids"],
            outputs=[SimpleNamespace(token_ids=[702, 703, 703], finish_reason="stop")],
        )]


def adapter(engine, *, generation_parser=parser, policy_version_getter=lambda: "old"):
    return VLLMGenerator(
        engine=engine,
        policy_version_getter=policy_version_getter,
        sampling_params_factory=lambda **kwargs: kwargs,
        sampling_settings={"n": 1, "temperature": 0.7, "top_p": 0.95, "max_tokens": 10},
        generation_parser=generation_parser,
        engine_generate_kwargs={"use_tqdm": False},
    )


def test_vllm_passes_native_prompt_seed_explicit_sampling_and_preserves_output():
    engine = FakeEngine()
    backend = adapter(engine)
    result = backend.generate((101, 104, 104), seed=23)
    assert result.token_ids == (702, 703, 703)
    assert engine.calls == [(
        [{"prompt_token_ids": [101, 104, 104]}],
        {"n": 1, "temperature": 0.7, "top_p": 0.95, "max_tokens": 10, "seed": 23},
        {"use_tqdm": False},
    )]


def test_vllm_rejects_parser_that_rewrites_native_output():
    backend = adapter(FakeEngine(), generation_parser=lambda ids, reason: parser((1, 2), reason))
    with pytest.raises(MalformedGenerationError, match="changed native IDs"):
        backend.generate((100,), seed=1)


def test_vllm_rejects_inflight_policy_change():
    versions = iter(("old", "new"))
    backend = adapter(FakeEngine(), policy_version_getter=lambda: next(versions))
    with pytest.raises(PolicyVersionError):
        backend.generate((100,), seed=1)


def test_vllm_rejects_multiple_completions():
    class MultiEngine(FakeEngine):
        def generate(self, **kwargs):
            output = super().generate(**kwargs)
            output[0].outputs *= 2
            return output

    with pytest.raises(BackendContractError, match="exactly one"):
        adapter(MultiEngine()).generate((100,), seed=1)


def entropy_trajectory():
    return Trajectory("query", "trajectory", "old", (1, 2, 3, 4, 1),
                      (False, True, True, False, True), (), 0.0, "test")


def test_transformer_entropy_is_full_vocabulary_causally_aligned_and_chunk_invariant():
    torch = pytest.importorskip("torch")

    class CachedCausalModel:
        training = False

        def __init__(self):
            self.calls = []

        def __call__(self, *, input_ids, attention_mask, past_key_values,
                     use_cache, return_dict):
            prior = () if past_key_values is None else past_key_values
            incoming = tuple(input_ids[0].tolist())
            self.calls.append((prior, incoming, tuple(attention_mask.shape)))
            total = sum(prior)
            rows = []
            for token in incoming:
                total += token
                rows.append(torch.tensor([0.0, float(total), -1.0, 0.5]))
            return SimpleNamespace(logits=torch.stack(rows).unsqueeze(0),
                                   past_key_values=prior + incoming)

    def compute(chunk_size):
        model = CachedCausalModel()
        provider = TransformersEntropyProvider(
            model=model, policy_version_getter=lambda: "old", device="cpu",
            chunk_size=chunk_size, max_context_tokens=20, model_call_kwargs={},
        )
        return provider.token_entropies(entropy_trajectory()), model.calls

    small, calls = compute(2)
    whole, _ = compute(4)
    assert isnan(small[0])
    assert small[1:] == pytest.approx(whole[1:])
    assert len(small) == 5
    assert calls == [((), (1, 2), (1, 2)), ((1, 2), (3, 4), (1, 4))]
    expected = []
    for preceding_sum in (1, 3, 6, 10):
        logits = torch.tensor([0.0, float(preceding_sum), -1.0, 0.5])
        log_probs = torch.log_softmax(logits, dim=-1)
        expected.append(float(-(log_probs.exp() * log_probs).sum()))
    assert small[1:] == pytest.approx(expected)


def test_entropy_rejects_wrong_policy_before_importing_or_calling_model():
    provider = TransformersEntropyProvider(
        model=SimpleNamespace(training=False), policy_version_getter=lambda: "new",
        device="cpu", chunk_size=2, max_context_tokens=20, model_call_kwargs={},
    )
    with pytest.raises(PolicyVersionError):
        provider.token_entropies(entropy_trajectory())


def test_entropy_requires_eval_mode_instead_of_mutating_the_callers_model():
    model = SimpleNamespace(training=True)
    provider = TransformersEntropyProvider(
        model=model, policy_version_getter=lambda: "old", device="cpu",
        chunk_size=2, max_context_tokens=20, model_call_kwargs={},
    )
    with pytest.raises(BackendContractError, match="eval mode"):
        provider.token_entropies(entropy_trajectory())
    assert model.training


@pytest.mark.parametrize("argument", ["position_ids", "cache_position", "inputs_embeds", "logits_to_keep"])
def test_entropy_rejects_kwargs_that_override_native_causal_alignment(argument):
    with pytest.raises(ValueError, match="alignment argument"):
        TransformersEntropyProvider(
            model=SimpleNamespace(training=False), policy_version_getter=lambda: "old",
            device="cpu", chunk_size=2, max_context_tokens=20,
            model_call_kwargs={argument: object()},
        )


def test_entropy_requires_actual_boolean_mask_elements():
    from dataclasses import replace

    source = entropy_trajectory()
    source = replace(source, generated_mask=(False, 2, True, False, True))
    provider = TransformersEntropyProvider(
        model=SimpleNamespace(training=False), policy_version_getter=lambda: "old",
        device="cpu", chunk_size=2, max_context_tokens=20, model_call_kwargs={},
    )
    with pytest.raises(BackendContractError, match="boolean"):
        provider.token_entropies(source)

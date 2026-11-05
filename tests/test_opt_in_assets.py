"""Opt-in acceptance cases. No default local paths, downloads or generated expectations."""

import hashlib
import json
import math
import os

import pytest

from ricepo.artifacts import file_sha256, require_cluster_storage, require_slurm
from ricepo.asset_identity import local_model_identity
from ricepo.backends import TransformersEntropyProvider
from ricepo.contracts import RankedDocument, Span, Trajectory
from ricepo.environment import configure_caches
from ricepo.formats import PromptRenderer
from ricepo.rollout import MalformedGenerationError


def _case(variable):
    filename, pinned_sha256 = os.environ.get(variable), os.environ.get(f"{variable}_SHA256")
    if filename is None and pinned_sha256 is None:
        pytest.skip(f"no explicitly supplied {variable} and independent SHA-256")
    if not filename or not pinned_sha256:
        pytest.fail(f"{variable} and {variable}_SHA256 must be supplied together")
    require_slurm()
    payload = require_cluster_storage(filename).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == pinned_sha256
    case = json.loads(payload)
    assert type(case["schema_version"]) is int and case["schema_version"] == 1
    configure_caches(case["cache_root"])
    return case


def test_opt_in_actual_tokenizer_template_native_boundaries():
    case = _case("RICEPO_TOKENIZER_FIXTURE")
    from transformers import AutoTokenizer

    snapshot = require_cluster_storage(case["tokenizer_directory"])
    files = {}
    for entry in snapshot.rglob("*"):
        if entry.is_symlink() and entry.is_dir():
            pytest.fail("use a concrete tokenizer-only directory without directory symlinks")
        if entry.is_file():
            assert entry.suffix not in {".bin", ".safetensors"}, "provide a tokenizer-only snapshot"
            files[entry.relative_to(snapshot).as_posix()] = file_sha256(require_cluster_storage(entry))
    assert files and files == case["file_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    prompt = require_cluster_storage(case["system_prompt"])
    assert file_sha256(prompt) == case["system_prompt_sha256"]
    assert tokenizer.chat_template and case["cases"]
    required = {"success", "failure"}
    for row in case["cases"]:
        renderer = PromptRenderer(tokenizer, prompt.read_text("utf-8"),
                                  observation_top_k=row["observation_top_k"],
                                  observation_token_limit=row["observation_token_limit"],
                                  prefill_think=row["prefill_think"])
        docs = [RankedDocument(d["id"], 0.0, d["text"]) for d in row["documents"]]
        assert renderer.initial(row["query"], docs) == tuple(row["initial_ids"])
        assert renderer.observation(docs) == tuple(row["observation_ids"])
        ids = tuple(row["completion_ids"])
        assert ids and all(type(i) is int and 0 <= i < len(tokenizer) for i in ids)
        parse = renderer.make_parser(malformed_policy="raise")
        if "error_code" in row:
            with pytest.raises(MalformedGenerationError) as caught:
                parse(ids, row["finish_reason"])
            assert caught.value.code == row["error_code"]
            required.discard("failure")
        else:
            result = parse(ids, row["finish_reason"])
            assert result.token_ids == ids
            assert result.reasoning_span == Span(*row["reasoning_span"])
            assert result.summary_span == Span(*row["summary_span"])
            assert result.summary_text == row["summary_text"]
            required.discard("success")
    assert not required, "supply at least one independently captured success and one failure case"


def test_opt_in_gpu_chunked_entropy_against_full_forward_on_same_native_ids():
    case = _case("RICEPO_GPU_ENTROPY_FIXTURE")
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device(case["device"])
    assert device.type == "cuda" and device.index is not None and torch.cuda.is_available()
    assert case["dtype"] in {"float32", "bfloat16"}
    dtype = getattr(torch, case["dtype"])
    model_dir = require_cluster_storage(case["model_directory"])
    identity = local_model_identity(model_dir)
    assert identity["payload_sha256"] == case["payload_sha256"]
    model, load_info = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
                trust_remote_code=False, torch_dtype=dtype, output_loading_info=True)
    assert not any(load_info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
    model.to(device).eval().requires_grad_(False)
    assert local_model_identity(model_dir) == identity and model.dtype == dtype
    ids, mask = tuple(case["token_ids"]), tuple(case["generated_mask"])
    assert 3 <= len(ids) == len(mask) <= 512 and type(case["chunk_size"]) is int
    assert 0 < case["chunk_size"] < len(ids) - 1 and mask[0] is False and any(mask)
    assert all(type(i) is int and 0 <= i < model.config.vocab_size for i in ids)
    assert all(type(v) is bool for v in mask)
    for key in ("atol", "rtol"):
        assert type(case[key]) in {int, float} and math.isfinite(case[key]) and case[key] >= 0
    version = "local-payload/" + identity["payload_sha256"]
    trajectory = Trajectory("fixture", "native", version, ids, mask, (), 0.0, "numerical_fixture")
    provider = TransformersEntropyProvider(model=model, policy_version_getter=lambda: version,
                     device=device, chunk_size=case["chunk_size"], max_context_tokens=len(ids), model_call_kwargs={})
    observed = provider.token_entropies(trajectory)
    assert len(observed) == len(ids) and math.isnan(observed[0])
    with torch.inference_mode():
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                       use_cache=False, return_dict=True).logits
        assert tuple(logits.shape) == (1, len(ids), model.config.vocab_size)
        log_prob = logits[0, :-1].float().log_softmax(-1)
        expected = -(log_prob.exp() * log_prob).sum(-1).cpu()
    torch.testing.assert_close(torch.tensor(observed[1:], dtype=torch.float32), expected,
                               atol=case["atol"], rtol=case["rtol"])

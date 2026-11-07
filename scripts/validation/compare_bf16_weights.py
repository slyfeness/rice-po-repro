"""Explicit, offline-only comparison of saved vLLM actions and HF weights.

Run under an authorized Slurm allocation. All inputs are local; this script
never generates, tokenizes, optimizes, downloads, or modifies source artifacts.
Per-token records and local paths in its output are PRIVATE evidence.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from ricepo.audit import fingerprint
from ricepo.precision_probe import probability_diagnostics, matched_behavior_gate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_saved_rows(selection, candidates, config):
    """Check the entire frozen cohort before loading a model; no text repair."""
    by_id = {row["request_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate saved request identity")
    rows = []
    for selected in selection["requests"]:
        row = by_id[selected["request_id"]]
        prefix, action = row["prefix_token_ids"], row["token_ids"]
        if not prefix or not action or any(type(x) is not int for x in prefix + action):
            raise ValueError("nonempty native integer token IDs required")
        if fingerprint(prefix) != row["prefix_sha256"]:
            raise ValueError("native prefix fingerprint mismatch")
        for key in ("prefix_sha256", "seed", "stage"):
            if selected[key] != row[key]:
                raise ValueError("frozen selection identity mismatch")
        if selected["prefix_tokens"] != len(prefix) or selected["action_tokens"] != len(action):
            raise ValueError("native sequence length mismatch")
        if len(action) != len(row["sampled_log_probs"]):
            raise ValueError("sampled log-prob/token alignment mismatch")
        if len(prefix) + len(action) > 32768:
            raise ValueError("frozen context budget exceeded")
        rows.append(row)
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate frozen cohort request")
    if len(rows) != config["expected_requests"] or sum(len(r["token_ids"]) for r in rows) != config["expected_sampled_tokens"]:
        raise ValueError("unexpected cohort size")
    return rows


def main(args):
    from ricepo.artifacts import atomic_json
    from ricepo.asset_identity import local_model_identity, package_versions
    from ricepo.environment import configure_caches

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    configure_caches(Path(args.cache))
    config = read(args.config)
    atomic_json(out / "PREREGISTRATION.json", config)
    selection = read(args.selection)
    paths = sorted(Path(args.responses).glob("*.json"))
    rows = validate_saved_rows(selection, [read(p) for p in paths], config)
    prior = read(args.prior_result)
    prior_csv = Path(args.prior_result).with_name("MATCHED_PRECISION_COMPARISON.csv")
    with prior_csv.open(newline="") as fd:
        old_rows = list(csv.DictReader(fd))
    if [r["request_id"] for r in old_rows] != [r["request_id"] for r in rows]:
        raise ValueError("A result is not from this exact ordered cohort")
    identity = local_model_identity(args.model)
    if identity != prior["model_identity"]:
        raise ValueError("model/tokenizer payload differs from A")
    benchmark = read(args.benchmark)
    if Path(benchmark["model"]).resolve() != Path(args.model).resolve() or benchmark["numeric_mode"] != "bf16_fast":
        raise ValueError("saved behavior model or precision mismatch")
    response_shas = {p.name: sha(p) for p in paths}
    binding = dict(selection_sha256=sha(args.selection), prior_result_sha256=sha(args.prior_result),
                   prior_csv_sha256=sha(prior_csv), benchmark_sha256=sha(args.benchmark),
                   response_sha256=response_shas, model_identity=identity,
                   scope="physical engine-resident weight digest was not exported by historical vLLM; local snapshot/cast identity only")
    atomic_json(out / "INPUT_BINDING.json", binding)

    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                local_files_only=True, trust_remote_code=False).eval().to("cuda:0")
    init_seconds = time.monotonic() - start
    assert all(p.dtype == torch.float32 for p in model.parameters())
    environment = dict(packages=package_versions(["torch", "transformers", "vllm", "flash-attn", "safetensors"]),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        hf_attention=model.config._attn_implementation,
        attention_class=type(model.model.layers[0].self_attn).__name__,
        vllm_attention=benchmark["runtime"]["attention_backend"],
        model_init_seconds=init_seconds, dropout_mode="eval", tf32=False,
        log_softmax_dtype="float32", causal_position="prefix_length-1 through full_length-2",
        mask="all native prefix is context only; each saved output token incl EOS is an action")
    atomic_json(out / "ENVIRONMENT.json", environment)
    all_sampled = [lp for r in rows for lp in r["sampled_log_probs"]]
    comparisons = [dict(configuration="A", evidence="reused Slurm175507 same cohort", **prior["overall"])]
    per_request = []

    def score(label):
        all_hf = []
        total_seconds = 0.0
        for row in rows:
            prefix, action = row["prefix_token_ids"], row["token_ids"]
            ids = prefix + action
            x = torch.tensor([ids[:-1]], device="cuda:0")
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.no_grad():
                output = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
                logits = output.logits[0, len(prefix)-1:len(ids)-1]
                expected = torch.bfloat16 if label == "B" else torch.float32
                if logits.dtype != expected or logits.shape[0] != len(action):
                    raise ValueError("unexpected logits dtype or causal alignment")
                lp = []
                for offset in range(0, len(action), 64):
                    dist = logits[offset:offset+64].float().log_softmax(-1)
                    target = torch.tensor(action[offset:offset+64], device="cuda:0")
                    lp.extend(dist.gather(1, target[:, None]).squeeze(1).cpu().tolist())
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            total_seconds += seconds
            diag = probability_diagnostics(row["sampled_log_probs"], lp, clip_epsilon=config["clip_epsilon"])
            per_request.append(dict(configuration=label, request_id=row["request_id"], stage=row["stage"],
                prefix_tokens=len(prefix), forward_seconds=seconds, **diag))
            atomic_json(out / f"{label}-{row['request_id']}.json", dict(request_id=row["request_id"],
                prefix_sha256=row["prefix_sha256"], action_token_ids=action,
                action_mask=[False]*len(prefix)+[True]*len(action), finish_reason=row["finish_reason"],
                sampled_log_probs=row["sampled_log_probs"], hf_log_probs=lp,
                errors=[c-s for c,s in zip(lp,row["sampled_log_probs"],strict=True)]))
            all_hf.extend(lp)
            atomic_json(out / f"PROGRESS-{label}-{row['request_id']}.json",
                        dict(configuration=label, checked=sum(r["configuration"] == label for r in per_request)))
            del x, output, logits
        return dict(configuration=label, evidence="executed same frozen cohort", forward_seconds=total_seconds,
                    **probability_diagnostics(all_sampled, all_hf, clip_epsilon=config["clip_epsilon"]))

    # C has no previous token-level evidence for this exact cohort. Run once,
    # before B, so a failed B immediately ends GPU work as requested.
    comparisons.append(score("C"))
    model.to(dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    # Independently check explicit BF16 loading equals FP32->BF16 for EVERY
    # parameter, not just names/dimensions. CPU reference freed before B.
    reference = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                    local_files_only=True, trust_remote_code=False).eval()
    digest = hashlib.sha256()
    compared = 0
    for (name, param), (other_name, other) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        value = param.detach().cpu()
        if name != other_name or other.dtype != torch.bfloat16 or not torch.equal(value, other):
            raise ValueError("BF16 load versus deterministic cast weight mismatch")
        digest.update(name.encode()+b"\0")
        digest.update(str(tuple(value.shape)).encode()+b"\0")
        digest.update(value.view(torch.uint16).numpy().tobytes())
        compared += value.numel()
    atomic_json(out / "BF16_WEIGHT_IDENTITY.json", dict(named_parameter_bf16_sha256=digest.hexdigest(),
        compared_parameter_elements=compared, all_parameters_equal=True,
        routes=["FP32 load then explicit model.to(torch.bfloat16)", "independent dtype=torch.bfloat16 load"],
        historical_vllm_device_weights_not_extracted=True))
    del reference
    comparisons.append(score("B"))
    by_label = {r["configuration"]: r for r in comparisons}
    b_pass = matched_behavior_gate(by_label["B"], config["acceptance"])
    # Failure here is a completed negative experiment, not infrastructure error.
    for row in comparisons:
        row["probability_gate_pass"] = matched_behavior_gate(row, config["acceptance"])
    fields = sorted(set().union(*(r.keys() for r in comparisons)))
    with (out / "BF16_WEIGHT_COMPARISON.csv").open("x", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(comparisons, key=lambda r:r["configuration"]))
    atomic_json(out / "PER_REQUEST.json", per_request)
    unchanged = all(sha(p) == response_shas[p.name] for p in paths)
    if not unchanged:
        raise ValueError("saved evidence changed during scoring")
    result = dict(status="INCONCLUSIVE" if b_pass else "FAIL", probability_gate_pass=b_pass,
        reason="PPO/update/sync validation still required" if b_pass else "explicit BF16 weights fail frozen probability thresholds",
        comparisons=by_label, input_unchanged=unchanged, checked_requests=len(rows),
        sampled_tokens=len(all_sampled), optimizer_step_executed=False, generated_tokens=0,
        later_phases="pending gated validation" if b_pass else "not run: mandatory stop after probability FAIL",
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(out / "RESULT.json", result)
    print(json.dumps(dict(status=result["status"], probability_gate_pass=b_pass, B=by_label["B"], result_sha256=sha(out/"RESULT.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "responses", "prior-result", "benchmark", "model", "output", "cache", "config"):
        parser.add_argument("--"+name, required=True)
    main(parser.parse_args())

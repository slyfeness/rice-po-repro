"""Read-only evaluation of a development-selected checkpoint on a held-out split."""

import json
import os
from pathlib import Path
import random

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm, stage_receipt
from .config import require_enabled
from .data import BRIGHT_TEST_COUNTS, validate_split_manifests, verify_counts
from .evaluation import summarize
from .formats import PromptRenderer
from .hf_runtime import (
    HFNativeGenerator, _configure_caches, _domain_retriever, _local_file, _manifest,
    _model_source, _source_hashes,
)
from .retrieval import DiverEncoder, ExplicitQueryComposer, PyTrecReward
from .rollout import RolloutRules, RolloutRunner


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_checkpoint(config, deployment, *, checkpoint_dir, test_manifest, output_root) -> dict:
    """Evaluate one fixed checkpoint; never optimize or select on test results.

    Default full-BRIGHT mode requires all twelve paper domain counts (1,384
    queries). Matching counts do not prove unpublished split identity. An
    explicitly named partial diagnostic has distinct output labeling.
    """
    require_enabled(config)
    job_id = require_slurm()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("reference holdout supports a single process")
    train_root = require_cluster_storage(deployment["output_root"])
    checkpoint = require_cluster_storage(checkpoint_dir)
    output = require_cluster_storage(output_root)
    if output.is_relative_to(train_root) or output.exists():
        raise ValueError("holdout output must be a new directory outside the completed training run")
    completed = _json(train_root / "completed.json")
    chosen = completed["selected_checkpoint"]
    if checkpoint != require_cluster_storage(chosen["checkpoint"]) or not checkpoint.is_relative_to(train_root):
        raise ValueError("holdout may evaluate only the checkpoint already selected by the completed training run")
    receipt = _json(checkpoint / "complete.json")
    if receipt["policy_version"] != chosen["policy_version"]:
        raise ValueError("selected checkpoint policy identity disagrees with its completion receipt")
    actual_files = {str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*")
                    if path.is_file() and path != checkpoint / "complete.json"}
    if actual_files != set(receipt["files"]):
        raise ValueError("selected checkpoint file set differs from its completion receipt")
    for relative, digest in receipt["files"].items():
        if file_sha256(checkpoint / relative) != digest:
            raise ValueError(f"selected checkpoint file has changed: {relative}")
    if file_sha256(train_root / "config.json") != receipt["config_sha256"] or _json(train_root / "config.json") != config:
        raise ValueError("holdout must use the selected training run's unchanged research configuration")
    source_hashes = _source_hashes()
    if source_hashes != _json(train_root / "sources.json"):
        raise ValueError("holdout implementation differs from the selected training run")
    original_deployment = _json(train_root / "deployment.json")
    for key, value in original_deployment.items():
        if key not in {"test_scope", "expected_test_domains"} and deployment.get(key) != value:
            raise ValueError(f"holdout cannot silently change training deployment field {key}")
    settings = config["required_research_settings"]
    domain_specs = deployment["domains"]
    train_path = _local_file(deployment["train_manifest_path"], "train_manifest_path")
    dev_path = _local_file(deployment["development_manifest_path"], "development_manifest_path")
    test_path = _local_file(str(test_manifest), "test_manifest")
    inputs = _json(train_root / "run-identity.json")["input_hashes"]
    if file_sha256(train_path) != inputs["train"] or file_sha256(dev_path) != inputs["development"]:
        raise ValueError("training or development manifest changed after checkpoint selection")
    train_rows, _, _ = _manifest(train_path, set(domain_specs))
    dev_rows, _, _ = _manifest(dev_path, set(domain_specs))
    test_rows, examples, qrels = _manifest(test_path, set(domain_specs))
    validate_split_manifests({"train": train_rows, "development": dev_rows, "test": test_rows})
    scope = deployment.get("test_scope", "bright_full_12_domains")
    if scope == "bright_full_12_domains":
        expected = list(BRIGHT_TEST_COUNTS)
        verify_counts(examples.values(), BRIGHT_TEST_COUNTS)
        if len(examples) != 1384:
            raise ValueError("full BRIGHT holdout requires 1,384 test queries")
        label = "bright_full_12_domain_count_matched_manifest"
    elif scope == "partial_diagnostic":
        expected = deployment.get("expected_test_domains")
        if (not isinstance(expected, list) or not expected or len(set(expected)) != len(expected)
                or set(expected) != {example.domain for example in examples.values()}):
            raise ValueError("partial holdout requires its exact explicitly named expected_test_domains")
        label = "partial_diagnostic_not_full_bright"
    else:
        raise ValueError("test_scope must be bright_full_12_domains or partial_diagnostic")
    prompt_path = _local_file(deployment["prompt_path"], "prompt_path", large=False)
    if file_sha256(prompt_path) != inputs["prompt"]:
        raise ValueError("holdout prompt content differs from training")
    retriever_path = require_cluster_storage(deployment["retriever_model_path"])
    if _model_source(retriever_path) != _json(train_root / "model-sources.json")["retriever_model_path"]:
        raise ValueError("fixed retriever model/tokenizer assets changed after training")
    _configure_caches(require_cluster_storage(deployment["cache_root"]))
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    actor_device = torch.device(deployment["actor_device"])
    retriever_device = torch.device(deployment["retriever_device"])
    if (actor_device.type != "cuda" or retriever_device.type != "cuda" or actor_device.index is None
            or retriever_device.index is None or actor_device == retriever_device
            or max(actor_device.index, retriever_device.index) >= torch.cuda.device_count()):
        raise ValueError("holdout requires the two distinct Slurm-visible devices of the training deployment")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "holdout-inputs.json", {
        "scope": label, "slurm_job_id": job_id, "checkpoint": str(checkpoint),
        "policy_version": receipt["policy_version"], "checkpoint_receipt_sha256": file_sha256(checkpoint / "complete.json"),
        "selection": chosen, "train_manifest_sha256": inputs["train"], "development_manifest_sha256": inputs["development"],
        "test_manifest": str(test_path), "test_manifest_sha256": file_sha256(test_path),
        "source_sha256": source_hashes, "expected_test_domains": expected,
    })
    local_args = dict(local_files_only=True, trust_remote_code=False,
                      cache_dir=str(require_cluster_storage(deployment["cache_root"]) / "transformers"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "actor"), **local_args)
    actor = AutoModelForCausalLM.from_pretrained(str(checkpoint / "actor"), torch_dtype=torch.float32,
                                                **local_args).to(actor_device).eval()
    actor.requires_grad_(False)
    retriever_tokenizer = AutoTokenizer.from_pretrained(str(retriever_path), **local_args)
    dtype_name = deployment["dtype"]
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("unsupported holdout precision")
    retriever = AutoModel.from_pretrained(str(retriever_path), torch_dtype=getattr(torch, dtype_name),
                                          **local_args).to(retriever_device).eval()
    retriever.requires_grad_(False)
    encoder = DiverEncoder(retriever, retriever_tokenizer, device=retriever_device,
                           max_length=deployment["retriever_max_length"], document_prefix=deployment["document_prefix"])
    renderer = PromptRenderer(tokenizer, prompt_path.read_text(encoding="utf-8"),
                              observation_top_k=settings["observation_top_k"],
                              observation_token_limit=settings["observation_token_limit"], prefill_think=True)
    generator = HFNativeGenerator(
        model=actor, tokenizer=tokenizer, device=actor_device, policy_version_getter=lambda: receipt["policy_version"],
        max_new_tokens=settings["max_new_tokens_per_step"], max_context_tokens=settings["max_context_tokens"],
        prefilled_think=True, precision_context=lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(dtype_name == "bfloat16")),
    )
    reward = PyTrecReward(qrels)
    composer = ExplicitQueryComposer(**settings["query_composition"])
    fixed_sources = _json(train_root / "retriever-sources.json")
    rows = []
    all_seeds = {}
    for domain in expected:
        backend, domain_receipt = _domain_retriever(domain, domain_specs[domain], deployment, settings, config["model"],
                                                   encoder, renderer, examples, qrels)
        if domain_receipt != fixed_sources[domain]:
            raise ValueError(f"holdout corpus, vectors, or fixed retrieval version differs from training: {domain}")
        runner = RolloutRunner(
            generator=generator, retriever=backend, reward_scorer=reward, query_composer=composer,
            rules=RolloutRules(frozenset({"stop"}), settings["max_new_tokens_per_step"], settings["max_context_tokens"]),
            should_stop=lambda steps: False, final_reward=lambda steps: steps[-1].local_reward,
            excluded_ids_by_query={key: ex.excluded_ids for key, ex in examples.items() if ex.domain == domain},
        )
        for identity, example in examples.items():
            if example.domain != domain:
                continue
            serial = len(rows)
            random_source = random.Random(f"holdout:{settings['training_seeds'][0]}:{identity}")
            seeds = tuple(random_source.randrange(2**63) for _ in range(5))
            all_seeds[identity] = seeds
            initial = backend.retrieve(example.query, excluded_ids=example.excluded_ids)
            trajectory = runner.run(
                query_id=identity, trajectory_id=f"test-{serial:06d}", original_query=example.query,
                initial_prefix_token_ids=renderer.initial(example.query, initial.documents), seeds=seeds,
                old_policy_version=receipt["policy_version"],
            )
            atomic_json(output / "trajectories" / f"{serial:06d}.json", {
                "initial_ranking": [(doc.document_id, doc.score) for doc in initial.documents],
                "trajectory": trajectory,
            })
            rows.append({"query_id": json.loads(identity)[1], "domain": domain, "ndcg_at_10": trajectory.final_reward})
    summary = {"scope": label, "policy_version": receipt["policy_version"], "test_queries": len(rows),
               "checkpoint": str(checkpoint), "aggregation": summarize(rows, expected_domains=expected)}
    artifacts = {"results-by-query.json": atomic_json(output / "results-by-query.json", rows),
                 "generation-seeds.json": atomic_json(output / "generation-seeds.json", all_seeds),
                 "holdout-summary.json": atomic_json(output / "holdout-summary.json", summary)}
    stage_receipt(output, "evaluation", policy_version=receipt["policy_version"], artifacts=artifacts)
    return summary

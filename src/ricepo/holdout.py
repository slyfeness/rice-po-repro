"""Standalone frozen baselines and read-only evaluation of selected checkpoints."""

import json
import os
from pathlib import Path

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm, stage_receipt
from .assessment import evaluate_queries
from .asset_identity import local_model_identity
from .audit import attempt, cost_report, fingerprint, operation
from .config import require_execution, resolve
from .data import BRIGHT_TEST_COUNTS, verify_counts
from .dataset_bundle import verify_bundle
from .environment import configure_caches
from .formats import CHAT_TURN, PromptRenderer, native_summary_mode, turn_protocol
from .hf_runtime import HFNativeGenerator, _domain_retriever, _local_file, _manifest, _source_hashes
from .policy_identity import identify
from .retrieval import DiverEncoder, ExplicitQueryComposer, PyTrecReward
from .rollout import RolloutRules, RolloutRunner
from .state import read_json, selected_so_far, verify_checkpoint


def _deploy_hash(deployment):
    return fingerprint({k: v for k, v in deployment.items() if k not in ("expected_test_domains", "test_scope")})


def _verify_binding(binding, **checks):
    for key, expected in checks.items():
        if binding.get(key) != expected:
            raise ValueError(f"selected run binding differs for {key}")


def _selected_state(config, deployment, checkpoint_dir, bundle):
    """Resolve the selected policy and verify its original training identity before opening test qrels."""
    train_root = require_cluster_storage(deployment["output_root"])
    checkpoint = require_cluster_storage(checkpoint_dir)
    completed = read_json(train_root / "completed.json")
    chosen = completed["selected_checkpoint"]
    binding = read_json(train_root / "binding.json")
    train = resolve(config, "train")
    current_sources = _source_hashes()
    _verify_binding(binding, schema_version=2, config_sha256=fingerprint(config),
                    recipe_fingerprint=train["fingerprint"], source_sha256=fingerprint(current_sources),
                    deployment_sha256=_deploy_hash(deployment), dataset_receipt_sha256=bundle.receipt_sha256,
                    manifest_sha256=bundle.manifest_sha256)
    if (read_json(train_root / "config.json") != config or read_json(train_root / "recipe.json") != train
            or read_json(train_root / "sources.json") != current_sources
            or _deploy_hash(read_json(train_root / "deployment.json")) != binding["deployment_sha256"]):
        raise ValueError("recorded source, recipe, configuration or deployment differs from the selected run binding")
    if chosen != selected_so_far(train_root, train["values"]["checkpoint_selection"]) or not chosen:
        raise ValueError("chosen checkpoint disagrees with the completed development selection")
    updates = completed.get("completed_updates")
    if (type(updates) is not int or updates <= 0 or updates != train["values"]["total_updates"]
            or completed.get("checkpoint_selection") != train["values"]["checkpoint_selection"]):
        raise ValueError("completed run budget or selection policy disagrees with training")
    terminal = read_json(train_root / f"iteration-{updates - 1:06d}" / "iteration_complete.receipt.json")
    if (terminal["policy_version"] != completed["last_policy_version"]
            or terminal["artifacts"]["selected_checkpoint_so_far"] != chosen):
        raise ValueError("completed run has no matching terminal iteration receipt")
    if (checkpoint != train_root / f"iteration-{chosen['iteration']:06d}" / "checkpoint"
            or checkpoint != require_cluster_storage(chosen["checkpoint"])):
        raise ValueError("holdout may only evaluate the selected checkpoint of this completed run")
    receipt = verify_checkpoint(checkpoint, fingerprint(binding))
    if receipt["policy_version"] != chosen["policy_version"] or receipt["iteration"] != chosen["iteration"]:
        raise ValueError("selected checkpoint receipt disagrees with development selection")
    previous_models = read_json(train_root / "model-sources.json")
    previous_retrievers = read_json(train_root / "retriever-sources.json")
    _verify_binding(binding, model_sha256=fingerprint(previous_models), retriever_sha256=fingerprint(previous_retrievers))
    return checkpoint, chosen, binding, previous_models, previous_retrievers


def _scope(deployment, examples, frozen_ids):
    ids = {tuple(json.loads(key)) for key in examples}
    if ids != set(frozen_ids):
        raise ValueError("test manifest IDs differ from the precommitted receipt")
    scope = deployment["test_scope"]
    if scope == "bright_full_12_domains":
        verify_counts(examples.values(), BRIGHT_TEST_COUNTS)
        if deployment.get("expected_test_domains") is not None:
            declared = deployment["expected_test_domains"]
            if not isinstance(declared, list) or sorted(declared) != sorted(BRIGHT_TEST_COUNTS):
                raise ValueError("full BRIGHT scope requires every domain exactly once")
        return examples, "complete", "bright_full_12_domain_count_matched_manifest"
    if scope != "partial_diagnostic":
        raise ValueError("test_scope must explicitly name full BRIGHT or partial_diagnostic")
    domains = deployment["expected_test_domains"]
    if (not isinstance(domains, list) or not domains or len(set(domains)) != len(domains)
            or not set(domains) <= {ex.domain for ex in examples.values()}):
        raise ValueError("partial diagnostics require distinct expected_test_domains in the frozen test set")
    return ({key: ex for key, ex in examples.items() if ex.domain in domains},
            "partial_diagnostic", "partial_diagnostic_not_full_bright")


def _prompt(settings, deployment):
    path = _local_file(deployment["prompt_path"], "prompt_path", large=False)
    declared = Path(settings["prompt_artifact"])
    if not declared.is_absolute():
        declared = Path(__file__).resolve().parents[2] / declared
    if not declared.is_file() or file_sha256(declared) != file_sha256(path):
        raise ValueError("deployment prompt differs from resolved research prompt")
    return path


def _actor_runtime(path, tokenizer_path, prompt, settings, deployment, policy_version_getter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(deployment["actor_device"])
    fixed_device = torch.device(deployment["retriever_device"])
    if (device.type != "cuda" or device.index is None or fixed_device.type != "cuda"
            or fixed_device.index is None or device == fixed_device
            or max(device.index, fixed_device.index) >= torch.cuda.device_count()):
        raise ValueError("generated evaluation requires distinct Slurm-visible actor and retriever CUDA devices")
    precision = deployment["dtype"]
    if precision not in ("float32", "bfloat16"):
        raise ValueError("unsupported reference precision")
    before = local_model_identity(path)
    local = dict(local_files_only=True, trust_remote_code=False)
    # Training uses the original tokenizer for every policy version and recovery.
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), **local)
    actor, info = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32,
                                                      output_loading_info=True, **local)
    if (any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
            or local_model_identity(path) != before or tokenizer.eos_token_id is None):
        raise ValueError("incomplete or changing local policy/tokenizer payload")
    actor.to(device).eval()
    actor.requires_grad_(False)
    renderer = PromptRenderer(tokenizer, prompt.read_text(encoding="utf-8"),
                              observation_top_k=settings["observation_top_k"],
                              observation_token_limit=settings["observation_token_limit"], prefill_think=True)
    generator = HFNativeGenerator(
        model=actor, tokenizer=tokenizer, device=device, policy_version_getter=policy_version_getter,
        max_new_tokens=settings["max_new_tokens_per_step"], max_context_tokens=settings["max_context_tokens"],
        prefilled_think=True, precision_context=lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(precision == "bfloat16")),
        summary_mode=native_summary_mode(deployment["output_parser_mode"]),
        structured_stop=turn_protocol(deployment) == CHAT_TURN)
    tokenizer_behavior = {"tokenizer_length": len(tokenizer), "special_ids": tokenizer.all_special_ids,
                          "chat_template_sha256": fingerprint(tokenizer.chat_template)}
    return renderer, generator, tokenizer_behavior


def _evaluate(config, deployment, *, command, output_root, checkpoint_dir=None, test_manifest=None):
    require_execution(config, command)
    job = require_slurm()
    if deployment.get("runtime") != "single_process_hf_reference" or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("offline evaluation requires the single-process reference deployment")
    report = resolve(config, command)
    settings, method = report["values"], report["method"]
    output = require_cluster_storage(output_root)
    if command == "evaluate" and output.is_relative_to(require_cluster_storage(deployment["output_root"])):
        raise ValueError("holdout output must be a new directory outside the selected training run")
    output.mkdir(parents=True, exist_ok=False)
    with attempt(output):
        with operation("asset_binding", method=method, split="test"):
            configure_caches(deployment["cache_root"])
            bundle = verify_bundle(deployment)
            path = bundle.paths["test"]
            if test_manifest is not None and require_cluster_storage(test_manifest) != path:
                raise ValueError("test manifest differs from the path fixed by the precommitted dataset receipt")
            checkpoint = chosen = binding = previous_models = previous_retrievers = None
            if command == "evaluate":
                checkpoint, chosen, binding, previous_models, previous_retrievers = _selected_state(
                    config, deployment, checkpoint_dir, bundle)
            generate = method != "retrieval_only"
            asset_keys = ("actor_model_path", "retriever_model_path") if generate else ("retriever_model_path",)
            models = {key: local_model_identity(require_cluster_storage(deployment[key])) for key in asset_keys}
            if previous_models is not None and previous_models != models:
                raise ValueError("initial model/tokenizer or fixed retriever payload changed after training")
            prompt = _prompt(settings, deployment) if generate else None
            if binding is not None:
                _verify_binding(binding, prompt_sha256=file_sha256(prompt))
            # Formal gold labels are accessed only inside this held-out assessment.
            _, examples, qrels = _manifest(path, set(deployment["domains"]),
                                           expected_sha256=bundle.manifest_sha256["test"])
            selected_examples, scope, label = _scope(deployment, examples, bundle.identities["test"])
            encoder = DiverEncoder.from_local(deployment)
            if encoder.identity["assets"] != models["retriever_model_path"]:
                raise ValueError("deployed retriever asset identity changed during evaluation setup")
            if chosen:
                policy_version = chosen["policy_version"]
            elif generate:
                policy_version = "initial-actor/" + models["actor_model_path"]["payload_sha256"]
            else:
                policy_version = "fixed-retriever/" + models["retriever_model_path"]["payload_sha256"]
            renderer = generator = None
            behavior = None
            if generate:
                tokenizer_path = require_cluster_storage(deployment["actor_model_path"])
                policy_path = checkpoint / "actor" if checkpoint else tokenizer_path
                renderer, generator, tokenizer_behavior = _actor_runtime(
                    policy_path, tokenizer_path, prompt, settings, deployment, lambda: policy_version)
                behavior = dict(**tokenizer_behavior, root_seed=settings["training_seeds"][0],
                                temperature=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"],
                                repetition_penalty=1.0, model_mode="eval", precision=deployment["dtype"],
                                initial_actor_payload=models["actor_model_path"]["payload_sha256"],
                                prefilled_think=True, rng="sha256-json-stream-v1")
                behavior["output_parser_mode"] = deployment["output_parser_mode"]
                if turn_protocol(deployment) == CHAT_TURN:
                    behavior["turn_protocol"] = CHAT_TURN
                if local_model_identity(tokenizer_path) != models["actor_model_path"]:
                    raise ValueError("initial actor/tokenizer identity changed while loading the policy")
                live_version, live_identity = identify(generator.model, behavior)
                if binding:
                    _verify_binding(binding, behavior=behavior)
                    registry = require_cluster_storage(deployment["output_root"]) / "behavior-identities"
                    if live_version != chosen["policy_version"] or read_json(registry / f"{live_version}.json") != live_identity:
                        raise ValueError("loaded checkpoint has a different live behavior-policy identity")
                    verify_checkpoint(checkpoint, fingerprint(binding))
                policy_version = live_version
                atomic_json(output / "behavior-policy.json", live_identity)
            reward = PyTrecReward(qrels)
            retrievers, runners, domain_receipts = {}, {}, {}
            for domain, spec in deployment["domains"].items():
                backend, domain_receipts[domain] = _domain_retriever(
                    domain, spec, deployment, settings, config["model"], encoder, renderer, examples, qrels)
                retrievers[domain] = backend
                if generate:
                    runners[domain] = RolloutRunner(
                        generator=generator, retriever=backend, reward_scorer=reward,
                        render_next_turn=renderer.next_user_turn if turn_protocol(deployment) == CHAT_TURN else None,
                        query_composer=ExplicitQueryComposer(**settings["query_composition"]),
                        rules=RolloutRules(frozenset({"stop"}), settings["max_new_tokens_per_step"],
                                           settings["max_context_tokens"], depth=settings["rollout_depth"]),
                        should_stop=lambda steps: False, final_reward=lambda steps: steps[-1].local_reward,
                        excluded_ids_by_query={key: ex.excluded_ids for key, ex in examples.items() if ex.domain == domain})
            if previous_retrievers is not None and previous_retrievers != domain_receipts:
                raise ValueError("fixed retrieval assets or retrieval behavior changed after checkpoint selection")
            evidence = dict(scope=label, split="test", slurm_job_id=job, method=method, policy_version=policy_version,
                            source_sha256=_source_hashes(), config_sha256=fingerprint(config), recipe=report,
                            dataset_receipt_sha256=bundle.receipt_sha256, manifest_sha256=bundle.manifest_sha256,
                            models=models, retrievers=domain_receipts, deployment_sha256=_deploy_hash(deployment),
                            prompt_sha256=file_sha256(prompt) if prompt else None, behavior=behavior)
            if checkpoint is not None:
                evidence.update(selection=chosen, training_binding_sha256=fingerprint(binding),
                                checkpoint_receipt_sha256=file_sha256(checkpoint / "complete.json"))
            atomic_json(output / "evaluation-inputs.json", evidence)
        aggregate = evaluate_queries(
            examples=selected_examples, expected_ids=bundle.identities["test"], split="test", phase="holdout",
            method=method, root_seed=settings["training_seeds"][0] if generate else None,
            policy_version=policy_version, rollout_depth=settings.get("rollout_depth"),
            retrievers=retrievers, runners=runners, renderer=renderer, reward=reward, scope=scope,
            manifest_sha256=bundle.manifest_sha256["test"],
            backend_fingerprints={domain: fingerprint(receipt) for domain, receipt in domain_receipts.items()},
            behavior_sha256=fingerprint(dict(policy_version=policy_version, behavior=behavior,
                                             prompt=evidence["prompt_sha256"], recipe=report["fingerprint"],
                                             source_sha256=fingerprint(evidence["source_sha256"]))),
            output=output / "evaluation")
        summary = dict(scope=label, method=method, policy_version=policy_version, aggregation=aggregate)
        if checkpoint is not None:
            summary["checkpoint"] = str(checkpoint)
        digest = atomic_json(output / "holdout-summary.json", summary)
        cost = atomic_json(output / "cost.json", cost_report(output))
        stage_receipt(output, "evaluation", policy_version=policy_version,
                      artifacts={"holdout-summary.json": digest, "cost.json": cost,
                                 "evaluation-inputs.json": file_sha256(output / "evaluation-inputs.json"),
                                 "evaluation/summary.json": file_sha256(output / "evaluation/summary.json"),
                                 "evaluation/results-by-query.json": file_sha256(output / "evaluation/results-by-query.json")})
        return summary


def evaluate_baseline(config, deployment, *, output_root):
    """Score retrieval_only or the original prompting actor without training state."""
    return _evaluate(config, deployment, command="baseline", output_root=output_root)


def evaluate_checkpoint(config, deployment, *, checkpoint_dir, test_manifest, output_root):
    """Score only the policy already chosen from the completed training run's development results."""
    return _evaluate(config, deployment, command="evaluate", checkpoint_dir=checkpoint_dir,
                     test_manifest=test_manifest, output_root=output_root)

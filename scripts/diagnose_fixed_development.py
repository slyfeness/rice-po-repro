"""Slurm-only fixed-development format sweep using one seed schedule and archived failures."""
import os
from pathlib import Path
import torch
from ricepo.artifacts import atomic_json, file_sha256, require_slurm
from ricepo.audit import attempt, operation, cost_report, stable_seed
from ricepo.config import resolve
from ricepo.dataset_bundle import verify_bundle
from ricepo.hf_runtime import _manifest, _domain_retriever
from ricepo.holdout import _actor_runtime
from ricepo.policy_identity import identify
from ricepo.retrieval import DiverEncoder, PyTrecReward, ExplicitQueryComposer
from ricepo.rollout import RolloutRunner, RolloutRules, MalformedGenerationError
from ricepo.audit import GenerationTruncated
from ricepo.state import writer, read_json

def main():
    require_slurm()
    project = Path(os.environ["RICE_PROJECT"])
    prior = Path(os.environ["RICE_PRIOR_PROMPT_RUN"])
    metadata = read_json(Path(os.environ["RICE_SOURCE_CONFIG"]))
    dep, config = metadata["deployment"], metadata["config"]
    plan = resolve(config, "baseline")["values"]
    inputs = read_json(prior / "evaluation-inputs.json")
    policy = read_json(prior / "behavior-policy.json")
    tag = "fixed-dev-format-" + os.environ["SLURM_JOB_ID"]
    root = project / "runs" / tag
    with writer(root, resume=False), attempt(root):
        atomic_json(root / "source.json", dict(source=prior.name, config=metadata,
            evaluation_inputs_sha256=file_sha256(prior / "evaluation-inputs.json")))
        try:
            frozen = verify_bundle(dep)
            if inputs["manifest_sha256"]["development"] != frozen.manifest_sha256["development"]:
                raise ValueError("development manifest identity differs")
            _, examples, qrels = _manifest(frozen.paths["development"], set(dep["domains"]),
                expected_sha256=frozen.manifest_sha256["development"])
            if {tuple(__import__("json").loads(q)) for q in examples} != frozen.identities["development"]:
                raise ValueError("development query membership differs")
            failure_path, = prior.glob("attempts/*/failure.json")
            saved_failure = read_json(failure_path)
            failed_query = saved_failure["operation"]["query_id"]
            if saved_failure["category"] != "tag_structure":
                raise ValueError("unexpected archived failure type")
            renderer, generator, _ = _actor_runtime(Path(dep["actor_model_path"]), Path(dep["actor_model_path"]),
                Path(dep["prompt_path"]), plan, dep, lambda: version)
            version, live = identify(generator.model, inputs["behavior"])
            if live != policy or version != inputs["policy_version"]:
                raise ValueError("live actor differs from the original development actor")
            encoder = DiverEncoder.from_local(dep)
            reward = PyTrecReward(qrels)
            runners, retrievers = {}, {}
            for domain, spec in dep["domains"].items():
                backend, _ = _domain_retriever(domain, spec, dep, plan, config["model"], encoder,
                                               renderer, examples, qrels)
                retrievers[domain] = backend
                runners[domain] = RolloutRunner(generator=generator, retriever=backend, reward_scorer=reward,
                    query_composer=ExplicitQueryComposer(**plan["query_composition"]),
                    rules=RolloutRules(frozenset({"stop"}), plan["max_new_tokens_per_step"],
                                       plan["max_context_tokens"], 5),
                    should_stop=lambda steps: False, final_reward=lambda steps: steps[-1].local_reward,
                    render_next_turn=renderer.next_user_turn,
                    excluded_ids_by_query={q: ex.excluded_ids for q,ex in examples.items() if ex.domain==domain})
            rows = []
            for serial, key in enumerate(sorted(examples)):
                ex = examples[key]
                domain, original_id = __import__("json").loads(key)
                if key == failed_query:
                    record = dict(query=key, status="failed", archived=True,
                        category=saved_failure["category"], depth=saved_failure["operation"]["depth"],
                        failure_sha256=file_sha256(failure_path))
                else:
                    seeds = tuple(stable_seed(plan["training_seeds"][0], "evaluation", "development",
                                              domain, original_id, depth) for depth in range(1,6))
                    try:
                        with operation("format_query", split="development", query_id=key):
                            initial = retrievers[domain].retrieve(ex.query, excluded_ids=ex.excluded_ids)
                            atomic_json(root / "initial" / f"{serial:03d}.json", initial)
                            trajectory = runners[domain].run(query_id=key, trajectory_id=f"development-q{serial:06d}",
                                original_query=ex.query, initial_prefix_token_ids=renderer.initial(ex.query,initial.documents),
                                seeds=seeds, old_policy_version=version)
                        atomic_json(root / "trajectories" / f"{serial:03d}.json", trajectory)
                        record = dict(query=key, status="complete", final_ndcg=trajectory.final_reward,
                                      depths=len(trajectory.steps), seeds=seeds)
                    except (MalformedGenerationError, GenerationTruncated) as exc:
                        identity = getattr(exc, "audit_operation", {})
                        record = dict(query=key, status="failed", category=getattr(exc,"code","generation_length"),
                                      depth=identity.get("depth"), exception=type(exc).__name__, reason=str(exc),
                                      seeds=seeds)
                rows.append(record)
                atomic_json(root / "queries" / f"{serial:03d}.json", record)
                print("query", serial+1, len(examples), record["status"], flush=True)
            summary = dict(kind="fixed_development_format_diagnostic", expected=len(examples), processed=len(rows),
                complete=sum(row["status"]=="complete" for row in rows),
                failed=sum(row["status"]=="failed" for row in rows), formal_ndcg=None,
                archived_failures=1, records=rows)
            atomic_json(root / "summary.json", summary)
        finally:
            atomic_json(root / "cost.json", cost_report(root))
            atomic_json(root / "memory.json", [torch.cuda.max_memory_allocated(i)
                                              for i in range(torch.cuda.device_count())])


if __name__ == "__main__":
    main()

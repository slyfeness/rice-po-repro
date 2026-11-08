"""Eight fixed training queries, one fixed seed stream, two predeclared prompts."""
import os, sys, json
from pathlib import Path
import torch
from ricepo.artifacts import atomic_json, file_sha256, require_slurm
from ricepo.audit import attempt, operation, stable_seed, fingerprint, cost_report
from ricepo.config import load_config, resolve
from ricepo.dataset_bundle import verify_bundle
from ricepo.hf_runtime import _manifest, _domain_retriever, _source_hashes
from ricepo.holdout import _actor_runtime, _prompt
from ricepo.policy_identity import identify
from ricepo.retrieval import DiverEncoder, PyTrecReward, ExplicitQueryComposer
from ricepo.rollout import RolloutRunner, RolloutRules, MalformedGenerationError
from ricepo.audit import GenerationTruncated
from ricepo.state import read_json, writer

def main():
    require_slurm()
    project=Path(os.environ["RICE_PROJECT"])
    release=Path(os.environ["RICE_SOURCE_RELEASE"])
    condition=sys.argv[1]
    filenames=dict(figure5="ricepo_system.txt",prefill_compat="ricepo_prefilled_think_smoke.txt")
    prompt=release/"prompts"/filenames[condition]
    metadata=read_json(Path(os.environ["RICE_SOURCE_CONFIG"]))
    dep=metadata["deployment"]
    dep["prompt_path"]=str(prompt)
    config=load_config(release/"configs/prompting.json")
    config["profile"]="ablation"
    config["deviations"]=dict(max_new_tokens_per_step=dict(value=4096,reason="Frozen diagnostic capacity."),
        native_span_parser=dict(value="configured_native_body_coverage_v1",reason="Same body coverage for both prompts."))
    if condition=="prefill_compat":
        config["deviations"]["prompt_artifact"]=dict(value="prompts/"+filenames[condition],
            reason="Explicit non-paper prefilled-think compatibility variant.")
    plan=resolve(config,"baseline")
    if plan["issues"]:
        raise ValueError(plan["issues"])
    p=plan["values"]
    _prompt(p,dep)
    root=project/"runs"/("training-prompt-"+condition+"-"+os.environ["SLURM_JOB_ID"])
    dep["cache_root"]=str(project/"cache"/root.name)
    with writer(root,resume=False),attempt(root):
        atomic_json(root/"deployment.json",dep)
        atomic_json(root/"recipe.json",plan)
        atomic_json(root/"sources.json",_source_hashes())
        try:
            frozen=verify_bundle(dep)
            _, examples,qrels=_manifest(frozen.paths["train"],set(dep["domains"]),
                                        expected_sha256=frozen.manifest_sha256["train"])
            ordered=sorted(examples,key=lambda q:stable_seed(20261105,"protocol_training_selection_v1",q))[:8]
            if len(ordered)!=8:
                raise ValueError("fewer than eight frozen training queries")
            atomic_json(root/"precommitted-selection.json",dict(query_ids=ordered,
                rule="ascending stable_seed(20261105,protocol_training_selection_v1,query_id), first eight",
                seed_rule="stable_seed(42,training_prompt_pair_v1,domain,query_id,depth)",
                selection_reward_independent=True,condition=condition,prompt_sha256=file_sha256(prompt)))
            renderer,generator,tokenizer_meta=_actor_runtime(Path(dep["actor_model_path"]),Path(dep["actor_model_path"]),
                                                            prompt,p,dep,lambda:version)
            behavior=dict(**tokenizer_meta,condition=condition,prompt_sha256=file_sha256(prompt),
                          parser=dep["parser_settings"],protocol=dep["turn_protocol"],temperature=1.,top_p=1.,top_k=0,
                          model_mode="eval",dtype=dep["dtype"])
            version,identity=identify(generator.model,behavior)
            atomic_json(root/"policy.json",identity)
            encoder=DiverEncoder.from_local(dep)
            reward=PyTrecReward(qrels)
            retrievers,runners={},{}
            for domain,spec in dep["domains"].items():
                backend,evidence=_domain_retriever(domain,spec,dep,p,config["model"],encoder,renderer,examples,qrels)
                retrievers[domain]=backend
                atomic_json(root/("retriever-"+domain+".json"),evidence)
                runners[domain]=RolloutRunner(generator=generator,retriever=backend,reward_scorer=reward,
                    query_composer=ExplicitQueryComposer(**p["query_composition"]),
                    rules=RolloutRules(frozenset({"stop"}),p["max_new_tokens_per_step"],p["max_context_tokens"],5),
                    should_stop=lambda steps:False,final_reward=lambda steps:steps[-1].local_reward,
                    render_next_turn=renderer.next_user_turn,
                    excluded_ids_by_query={q:x.excluded_ids for q,x in examples.items() if x.domain==domain})
            rows=[]
            for serial,key in enumerate(ordered):
                ex=examples[key]
                seeds=tuple(stable_seed(42,"training_prompt_pair_v1",ex.domain,key,depth) for depth in range(1,6))
                try:
                    with operation("protocol_query",query_id=key,condition=condition,split="train"):
                        with operation("retrieval_D0",query_id=key):
                            initial=retrievers[ex.domain].retrieve(ex.query,excluded_ids=ex.excluded_ids)
                        atomic_json(root/"initial"/f"{serial:02d}.json",initial)
                        trajectory=runners[ex.domain].run(query_id=key,trajectory_id=f"protocol-q{serial}",
                            original_query=ex.query,initial_prefix_token_ids=renderer.initial(ex.query,initial.documents),
                            seeds=seeds,old_policy_version=version)
                    atomic_json(root/"trajectories"/f"{serial:02d}.json",trajectory)
                    row=dict(query=key,status="complete",final_reward=trajectory.final_reward)
                except (MalformedGenerationError,GenerationTruncated) as exc:
                    row=dict(query=key,status="failed",category=getattr(exc,"code","generation_length"),
                             reason=str(exc),depth=getattr(exc,"audit_operation",{}).get("depth"))
                atomic_json(root/"queries"/f"{serial:02d}.json",dict(**row,seeds=seeds))
                rows.append(row)
                print(condition,serial+1,row["status"],flush=True)
            atomic_json(root/"summary.json",dict(condition=condition,queries=8,
                        complete=sum(row["status"]=="complete" for row in rows),records=rows))
        finally:
            atomic_json(root/"cost.json",cost_report(root))
            atomic_json(root/"memory.json",[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])


if __name__ == "__main__":
    main()

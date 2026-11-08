"""Same workload one/two engines, real retrieval, then immutable snapshot switch."""
import json,os,time
from pathlib import Path
R=Path(os.environ['RICE_PROJECT_ROOT']);B=Path(os.environ['RICE_BENCHMARK_ROOT'])
read=lambda p:json.loads(p.read_text())

def main():
    from ricepo.environment import configure_caches
    from ricepo.artifacts import atomic_json
    from ricepo.inference_pool import InferenceWorker
    from ricepo.inference_scheduling import partition_requests,parallel_conditions
    from ricepo.inference_retrieval_probe import load_retrieval,retrieval_workload
    from ricepo.audit import attempt,stable_seed,fingerprint
    from ricepo.vllm_two_stage import VLLMTwoStage
    from ricepo.rollout import RolloutRunner,RolloutRules,BranchRequest
    from ricepo.contracts import Anchor
    from ricepo.retrieval import ExplicitQueryComposer
    from transformers import AutoTokenizer
    O=B/f'dual-{os.environ["SLURM_JOB_ID"]}';O.mkdir();cache=R/'cache'/O.name;configure_caches(cache)
    m=read(B/'MANIFEST.json');root=Path(os.environ['RICE_PARENT_RUN']);masks=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(masks)!=2:raise ValueError('exactly two Slurm devices required')
    workers=[]
    def make(i,key,tag):
        return InferenceWorker(dict(cuda_mask=masks[i],manifest=str(B/'MANIFEST.json'),model_key=key,
            runtime=dict(gpu_memory_utilization=.45 if i else .80),co_resident=bool(i),
            rpc_root=str(Path(os.environ['RICE_RPC_ROOT'])/f'd{os.environ["SLURM_JOB_ID"]}{tag}{i}'),
            cache_root=str(cache/f'{tag}-{i}'),evidence=str(O/f'{tag}-worker-{i}')),wait=False)
    try:
        dep=read(root/'deployment.json');recipe=read(root/'recipe.json')['values'];binding=read(root/'binding.json')
        backends,reward,encoder,renderer,examples=load_retrieval(dep,recipe,binding,device='cuda:1',cache_root=cache)
        retrieval_workload(backends,reward,m['retrieval_requests'][:2],repetitions=1)
        workload=lambda:retrieval_workload(backends,reward,m['retrieval_requests'])
        baseline=workload();atomic_json(O/'RETRIEVAL_BASELINE.json',baseline)
        if not baseline['all_saved_ranks_equal'] or not baseline['all_saved_ndcg_equal']:raise ValueError('saved retrieval mismatch')
        rows=m['requests'];workers=[make(0,'initial','a')];workers[0].wait_ready()
        a=parallel_conditions(workers,partition_requests(rows,1),O/'single',retrieval=workload)
        atomic_json(O/'SINGLE.json',a)
        if not a['retrieval']['all_saved_ranks_equal'] or not a['retrieval']['all_saved_ndcg_equal']:raise ValueError('single baseline rank drift')
        workers.append(make(1,'initial','b'));workers[1].wait_ready()
        atomic_json(O/'INITIALIZATION.json',[w.ready for w in workers])
        results={}
        for state in ('cold','hot'):
            results[state]=parallel_conditions(workers,partition_requests(rows,2),O/('dual-'+state),
                retrieval=workload,reset_cache=state=='cold')
            rr=results[state]['retrieval']
            if not rr['all_saved_ranks_equal'] or not rr['all_saved_ndcg_equal']:raise ValueError('co-resident retrieval mismatch')
            atomic_json(O/('DUAL_'+state.upper()+'.json'),results[state])
        # Minimal real online protocol: first two fixed training identities, no selection by outcome.
        tok=AutoTokenizer.from_pretrained(m['models']['initial'],local_files_only=True,trust_remote_code=False)
        closed=[];tick=time.monotonic()
        with attempt(O/'closed-loop'):
            for i,qid in enumerate(sorted(examples)[:2]):
                ex=examples[qid];w=workers[i];sampler=VLLMTwoStage(w,tok,lambda w=w:w.ready['policy'],4096,32768)
                runner=RolloutRunner(generator=sampler,retriever=backends[ex.domain],reward_scorer=reward,
                    query_composer=ExplicitQueryComposer(**recipe['query_composition']),
                    rules=RolloutRules(frozenset({'stop'}),4096,32768,5),should_stop=lambda steps:False,
                    final_reward=lambda steps:steps[-1].local_reward,render_next_turn=renderer.next_user_turn,
                    failure_semantics='invalid_action_terminal_zero_v1',excluded_ids_by_query={qid:ex.excluded_ids})
                initial=backends[ex.domain].retrieve(ex.query,excluded_ids=ex.excluded_ids)
                trajectory=runner.run(query_id=qid,trajectory_id=f'schedule-{i}',original_query=ex.query,
                    initial_prefix_token_ids=renderer.initial(ex.query,initial.documents),
                    seeds=tuple(stable_seed(20261107,'closed',qid,d) for d in range(1,6)),
                    old_policy_version=w.ready['policy'],initial_ndcg=reward.score(qid,[d.document_id for d in initial.documents]))
                atomic_json(O/f'closed-trajectory-{i}.json',trajectory);closed.append(trajectory)
                if trajectory.steps:
                    s=trajectory.steps[0];anchor=Anchor(qid,trajectory.trajectory_id,1,0.,s.reasoning_span,s.summary_span)
                    # Fixed first actual Summary, not entropy selection; scheduling diagnostic only.
                    branches=runner.branch(original_query=ex.query,original=trajectory,anchor=anchor,
                        requests=tuple(BranchRequest(f'diagnostic-{i}-{b}',tuple(stable_seed(20261107,'branch',qid,b,d) for d in range(1,6))) for b in range(5)),batched=True)
                    atomic_json(O/f'closed-branch-{i}.json',branches)
        atomic_json(O/'CLOSED_LOOP.json',dict(seconds=time.monotonic()-tick,trajectories=closed,optimizer_steps=0,
            anchor_selection='first existing legal Summary; scheduling-only, no entropy/credit/PPO'))
        initial=[w.ready for w in workers]
        for w in workers:w.close()
        workers=[make(i,'updated','new') for i in range(2)]
        for w in workers:w.wait_ready()
        if workers[0].ready['policy']==initial[0]['policy']:raise ValueError('version did not change')
        probe=sum([[r for r in rows if r['stage']==s][:4] for s in ('think','summary')],[])
        switched=parallel_conditions(workers,partition_requests(probe,2),O/'updated-snapshot',cap=64)
        atomic_json(O/'VERSION_SWITCH.json',dict(old=initial,new=[w.ready for w in workers],probe=switched,
            kv_invalidation='old processes terminated; both new immutable engines constructed',optimizer_steps=0))
        atomic_json(O/'RESULT.json',dict(status='PASS',single=a,dual=results,optimizer_steps=0))
        print('DUAL_SCHEDULING_PASS',flush=True)
    except BaseException as e:
        atomic_json(O/'FAILURE.json',dict(type=type(e).__name__,reason=str(e),optimizer_steps=0));raise
    finally:
        for w in workers:w.close()

if __name__=='__main__':main()

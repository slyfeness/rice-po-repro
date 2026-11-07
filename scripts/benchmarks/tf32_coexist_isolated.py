"""One bounded process-isolation retry; no generation resampling or PPO."""
import json,os,time
from pathlib import Path
R=Path(os.environ['RICE_PROJECT_ROOT']);B=Path(os.environ['RICE_BENCHMARK_ROOT'])
read=lambda p:json.loads(p.read_text())

def main():
    from ricepo.environment import configure_caches
    from ricepo.artifacts import atomic_json
    from ricepo.inference_retrieval_probe import load_retrieval,retrieval_workload
    from ricepo.inference_pool import InferenceWorker
    import torch
    O=B/f'coexist-isolated-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
    cache=R/'cache'/O.name;configure_caches(cache)
    m=read(B/'MANIFEST.json');root=Path(os.environ['RICE_PARENT_RUN'])
    worker=None;started=time.monotonic()
    try:
        backends,reward,encoder,_,_=load_retrieval(read(root/'deployment.json'),read(root/'recipe.json')['values'],
            read(root/'binding.json'),device='cuda:0',cache_root=cache)
        torch.cuda.synchronize();diver=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        retrieval_workload(backends,reward,m['retrieval_requests'][:2],repetitions=1)
        baseline=retrieval_workload(backends,reward,m['retrieval_requests'])
        atomic_json(O/'RETRIEVAL_BASELINE.json',baseline)
        if not baseline['all_saved_ranks_equal'] or not baseline['all_saved_ndcg_equal']:raise ValueError('baseline rank drift')
        spec=dict(cuda_mask=os.environ['CUDA_VISIBLE_DEVICES'],manifest=str(B/'MANIFEST.json'),
            runtime=dict(gpu_memory_utilization=.45),co_resident=True,
            rpc_root=str(Path(os.environ['RICE_RPC_ROOT'])/f'c{os.environ["SLURM_JOB_ID"]}'),
            cache_root=str(cache/'worker'),evidence=str(O/'worker'))
        worker=InferenceWorker(spec);atomic_json(O/'INITIALIZATION.json',dict(diver=diver,worker=worker.ready))
        ids=sum([[r['request_id'] for r in m['requests'] if r['stage']==s][:144] for s in ('think','summary')],[])
        tick=time.monotonic();worker.start_conditions(ids,O/'generation',reset_cache=True)
        retrieved=retrieval_workload(backends,reward,m['retrieval_requests'])
        generated=worker.finish_conditions();wall=time.monotonic()-tick
        atomic_json(O/'RETRIEVAL_PARALLEL.json',retrieved)
        if not retrieved['all_saved_ranks_equal'] or not retrieved['all_saved_ndcg_equal']:raise ValueError('concurrent rank drift')
        atomic_json(O/'RESULT.json',dict(status='COEXISTENCE_PASS',diver=diver,worker=worker.ready,
            combined_wall_seconds=wall,generation=generated,
            retrieval_baseline_seconds=baseline['wall_seconds'],retrieval_parallel_seconds=retrieved['wall_seconds'],
            retrieval_interference_ratio=retrieved['wall_seconds']/baseline['wall_seconds'],
            actual_retrieval_requests=retrieved['requests'],ranks_and_ndcg_identical=True,optimizer_steps=0,
            kernel_overlap='NOT_MEASURED; overlapping host intervals do not prove kernel overlap'))
        print('ISOLATED_COEXISTENCE_PASS',wall,flush=True)
    except BaseException as e:
        atomic_json(O/'FAILURE.json',dict(type=type(e).__name__,reason=str(e),optimizer_steps=0));raise
    finally:
        if worker is not None:worker.close()

if __name__=='__main__':main()

"""Measured four-TP1 pure inference; a single shared coordinator clock."""
import json,os,time
from pathlib import Path
R=Path(os.environ['RICE_PROJECT_ROOT']);B=Path(os.environ['RICE_BENCHMARK_ROOT'])
def main():
    from ricepo.artifacts import atomic_json
    from ricepo.inference_pool import InferenceWorker
    from ricepo.inference_scheduling import partition_requests,parallel_conditions
    m=json.loads((B/'MANIFEST.json').read_text());O=B/f'four-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
    masks=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(masks)!=4:raise ValueError('four Slurm devices required')
    workers=[];started=time.monotonic()
    try:
        for i,mask in enumerate(masks):
            workers.append(InferenceWorker(dict(cuda_mask=mask,manifest=str(B/'MANIFEST.json'),runtime=m['candidates']['S2'],
                rpc_root=str(Path(os.environ['RICE_RPC_ROOT'])/f'f{os.environ["SLURM_JOB_ID"]}{i}'),
                cache_root=str(R/'cache'/O.name/str(i)),evidence=str(O/f'worker-{i}')),wait=False))
        for w in workers:w.wait_ready()
        ready=time.monotonic();atomic_json(O/'INITIALIZATION.json',dict(workers=[w.ready for w in workers],
            all_ready_seconds=ready-started))
        results={}
        for state in ('cold','hot'):
            results[state]=parallel_conditions(workers,partition_requests(m['requests'],4),O/state,reset_cache=state=='cold')
            atomic_json(O/(state.upper()+'.json'),results[state])
            print('FOUR_BATCH',state,results[state]['generated_tokens'],results[state]['common_wall_seconds'],
                results[state]['tokens_per_second'],flush=True)
        atomic_json(O/'RESULT.json',dict(status='PASS',candidate='S2',results=results,workers=[w.ready for w in workers],
            initialization_seconds=ready-started,initialization_plus_cold_seconds=ready-started+results['cold']['common_wall_seconds'],
            optimizer_steps=0,scope='pure inference, not four-GPU training'))
    except BaseException as e:
        atomic_json(O/'FAILURE.json',dict(type=type(e).__name__,reason=str(e)));raise
    finally:
        for w in workers:w.close()
if __name__=='__main__':main()

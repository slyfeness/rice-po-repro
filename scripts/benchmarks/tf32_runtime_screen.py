"""Fixed three-candidate inference-only screen; native cold/hot observations."""
import json,os,sys,time
from pathlib import Path
R=Path(os.environ['RICE_PROJECT_ROOT'])
B=Path(os.environ['RICE_BENCHMARK_ROOT'])
read=lambda p:json.loads(p.read_text())

def main():
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.environment import configure_caches
    from ricepo.asset_identity import local_model_identity
    from ricepo.inference_benchmark import Observations,run_conditions
    from ricepo.vllm_two_stage import VLLMEngine,VLLMTwoStage,BACKEND
    from ricepo.audit import attempt,stable_seed
    from transformers import AutoTokenizer
    candidate=sys.argv[1];scope=sys.argv[2] if len(sys.argv)>2 else 'screen'
    manifest=read(B/'MANIFEST.json');runtime=manifest['candidates'][candidate]
    if os.environ.get('RICE_SCREEN_LOW_MEMORY')=='1':runtime['gpu_memory_utilization']=.8
    tag=f'{candidate}-{scope}-{os.environ["SLURM_JOB_ID"]}'
    O=B/tag;O.mkdir()
    configure_caches(R/'cache'/('tf32-throughput-'+tag))
    model=manifest['models']['initial']
    identity=local_model_identity(model)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    n=manifest['screen']['requests_per_stage'] if scope=='screen' else manifest['full_pool']['requests_per_stage']
    rows=sum([[r for r in manifest['requests'] if r['stage']==s][:n] for s in ('think','summary')],[])
    engine=None;tick=time.monotonic()
    try:
        with attempt(O),Observations() as obs:
            engine=VLLMEngine(model,tok,identity['payload_sha256'],numeric_mode='tf32_pilot',runtime_options=runtime,collect_metrics=True)
            obs.engine=engine;init=time.monotonic()-tick
            atomic_json(O/'INITIALIZATION.json',dict(model_identity=identity,manifest_sha256=file_sha256(B/'MANIFEST.json'),
                candidate=candidate,runtime=engine.runtime_options,initialization_seconds=init,
                observation=obs.interval(tick,time.monotonic()),source_kind='expanded Git checkout + source file overlay; no archive'))
            for cache in ('cold','hot'):
                if cache=='cold':engine.llm.reset_prefix_cache()
                result=run_conditions(engine,rows,O/cache,cap=512,observations=obs)
                print('BATCH',json.dumps(dict(candidate=candidate,scope=scope,cache=cache,requests=result['requests'],
                    tokens=result['generated_tokens'],wall=result['wall_seconds'],tokens_per_second=result['tokens_per_second'])),flush=True)
            if scope=='confirm':
                think=[r for r in manifest['requests'] if r['stage']=='think'][:16]
                if any(stable_seed(r['root_seed'],BACKEND,'think')!=r['seed'] for r in think):
                    raise ValueError('source stage-seed identity changed')
                sampler=VLLMTwoStage(engine,tok,lambda:engine.version,4096,32768)
                before=time.monotonic();out=sampler.generate_many([tuple(r['prefix_token_ids']) for r in think],[r['root_seed'] for r in think])
                seconds=time.monotonic()-before
                atomic_json(O/'LONG_TWO_STAGE.json',dict(request_ids=[r['request_id'] for r in think],generations=out,
                    shared_model_budget=4096,root_seeds=[r['root_seed'] for r in think],
                    rng_namespace=BACKEND,numeric_mode='tf32_pilot',wall_seconds=seconds,
                    generated_tokens=sum(sum(g.action_mask) for g in out),same_source_rng_namespace=True))
            atomic_json(O/'COMPLETE.json',dict(status='COMPLETE',candidate=candidate,scope=scope,
                requests_per_stage=n,formal_complete=True,optimizer_steps=0))
    except BaseException as e:
        atomic_json(O/'FAILURE.json',dict(type=type(e).__name__,reason=str(e),candidate=candidate,scope=scope))
        raise
    finally:
        if engine is not None:engine.llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()

# Public driver: explicit Slurm execution only; no downloads.
import os,sys,json,time,subprocess
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);M=R/'benchmarks/tf32-swap-175674/MANIFEST.json'

def main():
    group=sys.argv[1]
    if group not in ('B','C'):raise ValueError('only frozen B/C groups')
    os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='high' if group=='B' else 'highest'
    from ricepo.environment import configure_caches
    from ricepo.asset_identity import local_model_identity
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.vllm_two_stage import VLLMEngine
    from ricepo.inference_benchmark import Observations,run_conditions
    from ricepo.audit import attempt
    from transformers import AutoTokenizer
    O=R/'benchmarks'/f'tf32-gemm-{group}-{os.environ["SLURM_JOB_ID"]}';O.mkdir();configure_caches(R/'cache'/O.name)
    m=json.loads(M.read_text());model=m['models']['initial'];identity=local_model_identity(model)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    runtime=dict(m['candidates']['S2']);mode='tf32_pilot' if group=='B' else 'bf16_fast'
    if group=='C':runtime['attention_backend']='FLASH_ATTN'
    engine=None
    try:
        with attempt(O),Observations() as obs:
            tick=time.monotonic();engine=VLLMEngine(model,tok,identity['payload_sha256'],numeric_mode=mode,
                runtime_options=runtime,collect_metrics=True,worker_extension_cls='ricepo.gemm_diagnostic.GemmInspection')
            obs.engine=engine;init=time.monotonic()-tick;state=engine.llm.collective_rpc('gemm_state')
            if state[0]['matmul_allow_tf32']!=(group=='B'):raise ValueError('wrong worker matmul mode')
            device=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
            atomic_json(O/'INITIALIZATION.json',dict(group=group,worker_state=state,device_metadata=device,
                initialization_seconds=init,model_identity=identity,manifest_sha256=file_sha256(M),runtime=engine.runtime_options,
                hf_training='NOT_LOADED; frontend flags not changed'))
            results={}
            for cache in ('cold','hot'):
                if cache=='cold':engine.llm.reset_prefix_cache()
                results[cache]=run_conditions(engine,m['requests'],O/cache,cap=512,observations=obs)
                print('GEMM_BATCH',group,cache,results[cache]['generated_tokens'],results[cache]['wall_seconds'],results[cache]['tokens_per_second'],flush=True)
            atomic_json(O/'RESULT.json',dict(status='COMPLETE',group=group,results=results,optimizer_steps=0))
    except BaseException as e:
        atomic_json(O/'FAILURE.json',dict(type=type(e).__name__,reason=str(e)));raise
    finally:
        if engine is not None:engine.llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()

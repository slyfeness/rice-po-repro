# Public driver: explicit Slurm execution only; no downloads.
import json,os,time,sys
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);M=R/'benchmarks/tf32-swap-175674/MANIFEST.json'

def main():
    from ricepo.environment import configure_caches
    from ricepo.asset_identity import local_model_identity
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.vllm_two_stage import VLLMEngine
    from ricepo.audit import attempt
    from transformers import AutoTokenizer
    child=len(sys.argv)>1
    O=Path(sys.argv[2]) if child else R/'benchmarks'/f'tf32-gemm-probe-{os.environ["SLURM_JOB_ID"]}'
    if not child:O.mkdir()
    m=json.loads(M.read_text());model=m['models']['initial'];configure_caches(R/'cache'/O.name)
    identity=local_model_identity(model);tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    row=next(r for r in m['requests'] if r['stage']=='think')
    for label,precision in ([('B','high')] if child else [('A','highest')]):
        configure_caches(R/'cache'/O.name/label)
        os.environ['VLLM_FLOAT32_MATMUL_PRECISION']=precision
        engine=None;tick=time.monotonic();output=O/label
        try:
            with attempt(output):
                engine=VLLMEngine(model,tok,identity['payload_sha256'],numeric_mode='tf32_pilot',
                    runtime_options=m['candidates']['S2'],worker_extension_cls='ricepo.gemm_diagnostic.GemmInspection',
                    profiler_config=dict(profiler='torch',torch_profiler_dir=str(output/'profiler'),
                        torch_profiler_use_gzip=False,torch_profiler_with_stack=False,
                        torch_profiler_record_shapes=True,ignore_frontend=True))
                state=engine.llm.collective_rpc('gemm_state')
                atomic_json(output/'STATE.json',dict(state=state,initialization_seconds=time.monotonic()-tick,
                    manifest_sha256=file_sha256(M),model_identity=identity))
                if state[0]['matmul_allow_tf32']!=(label=='B'):raise ValueError('worker did not apply requested GEMM precision')
                engine.llm.reset_prefix_cache();engine.llm.start_profile()
                result=engine.sample([dict(prefix=tuple(row['prefix_token_ids']),seed=row['seed'],budget=8)],'think',engine.version)
                engine.llm.stop_profile()
                atomic_json(output/'NATIVE.json',dict(request=row,outputs=result,optimizer_steps=0))
                print('PROFILE_COMPLETE',label,flush=True)
        finally:
            if engine is not None:engine.llm.llm_engine.engine_core.shutdown()
        # Keep A/B in separate fresh Python frontends as well as worker processes.
        if label=='A':
            import subprocess
            subprocess.run([sys.executable,__file__,'B',str(O)],check=True);break

if __name__=='__main__':main()

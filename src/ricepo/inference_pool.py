"""Experimental immutable TP1 process workers; no Trainer or optimizer integration.

Each worker has its own CUDA mask, statistics registry, engine and KV lifetime.
JVM/Retriever work stays in the coordinator, outside vLLM frontend processes.
"""
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time


def _serve(pipe,spec):
    os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES']=spec['cuda_mask']
    os.environ['TRITON_F32_DEFAULT']='tf32'
    os.environ['VLLM_RPC_BASE_PATH']=spec['rpc_root']
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ.pop('JAVA_TOOL_OPTIONS',None)
    Path(spec['rpc_root']).mkdir(parents=True,exist_ok=False)
    from .environment import configure_caches
    configure_caches(spec['cache_root'])
    from .asset_identity import local_model_identity
    from .vllm_two_stage import VLLMEngine
    from .inference_benchmark import Observations,run_conditions
    from .audit import attempt
    from .artifacts import atomic_json
    from transformers import AutoTokenizer
    import json,torch
    engine=None
    try:
        manifest=json.loads(Path(spec['manifest']).read_text())
        by_id={r['request_id']:r for r in manifest['requests']}
        model=manifest['models'][spec.get('model_key','initial')]
        identity=local_model_identity(model)
        tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
        with attempt(Path(spec['evidence'])),Observations() as obs:
            start=time.monotonic()
            engine=VLLMEngine(model,tok,identity['payload_sha256'],numeric_mode='tf32_pilot',
                runtime_options=spec['runtime'],collect_metrics=True,co_resident=spec.get('co_resident',False))
            obs.engine=engine
            pipe.send(('ready',dict(policy=engine.version,assets=identity,runtime=engine.runtime_options,
                gpu_name=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                initialization_seconds=time.monotonic()-start,
                observation=obs.interval(start,time.monotonic()),pid=os.getpid())))
            while True:
                command=pipe.recv()
                if command['kind']=='close':break
                if command['kind']=='conditions':
                    if command.get('reset_cache'):engine.llm.reset_prefix_cache()
                    rows=[by_id[k] for k in command['request_ids']]
                    began=time.monotonic()
                    result=run_conditions(engine,rows,command['output'],cap=command.get('cap',512),observations=obs)
                    pipe.send(('result',dict(summary=result,start=began,end=time.monotonic(),policy=engine.version)))
                elif command['kind']=='sample':
                    output=engine.sample(command['requests'],command['stage'],engine.version)
                    atomic_json(Path(spec['evidence'])/f'sample-{command["serial"]}.json',dict(
                        requests=command['requests'],stage=command['stage'],outputs=output,policy=engine.version))
                    pipe.send(('result',output))
                else:raise ValueError('unsupported inference command')
    except BaseException as e:
        import traceback
        try:pipe.send(('error',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())))
        except (BrokenPipeError,EOFError):pass
        raise
    finally:
        if engine is not None:engine.llm.llm_engine.engine_core.shutdown()
        pipe.close()


class InferenceWorker:
    def __init__(self,spec,*,wait=True):
        ctx=multiprocessing.get_context('spawn');self.pipe,child=ctx.Pipe()
        self.process=ctx.Process(target=_serve,args=(child,spec));self.process.start();child.close()
        self.lock=threading.Lock();self.serial=0;self.closed=False;self.ready=None
        if wait:self.wait_ready()
    def receive(self):
        if not self.pipe.poll(1200):raise TimeoutError('worker timeout; no generation retry')
        kind,value=self.pipe.recv()
        if kind=='error':raise RuntimeError('worker failure: '+str(value))
        return kind,value
    def wait_ready(self):
        try:
            kind,value=self.receive()
            if kind!='ready':raise RuntimeError('worker readiness mismatch')
            self.ready=value;return value
        except BaseException:self.close(abort=True);raise
    def start_conditions(self,request_ids,output,*,reset_cache=False,cap=512):
        self.pipe.send(dict(kind='conditions',request_ids=request_ids,output=str(output),reset_cache=reset_cache,cap=cap))
    def finish_conditions(self):
        kind,value=self.receive()
        if kind!='result':raise RuntimeError('worker result mismatch')
        return value
    def sample(self,requests,stage,policy):
        if policy!=self.ready['policy']:raise ValueError('stale inference weight version')
        with self.lock:
            self.serial+=1;self.pipe.send(dict(kind='sample',requests=requests,stage=stage,serial=self.serial))
            kind,value=self.receive()
            if kind!='result':raise RuntimeError('sample response mismatch')
            return value
    def close(self,abort=False):
        if self.closed:return
        try:
            if not abort:
                self.pipe.send(dict(kind='close'));self.process.join(120)
            if self.process.is_alive():
                try:
                    if os.getpgid(self.process.pid)==self.process.pid:os.killpg(self.process.pid,signal.SIGTERM)
                    else:self.process.terminate()
                except ProcessLookupError:pass
                self.process.join(10)
        finally:self.pipe.close();self.closed=True

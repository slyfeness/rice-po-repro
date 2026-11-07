"""Phase-separated vLLM worker: immutable snapshots, process-local KV caches.

Uses the Actor's assigned GPU only while HF tensors are offloaded. Retrieval
keeps its existing second device. This is not a new four-rank topology.
"""
import multiprocessing
import os
import time
import signal
import tempfile
from pathlib import Path
from .audit import event,fingerprint
from .artifacts import atomic_json
from .vllm_two_stage import VLLMTwoStage,BACKEND,vllm_runtime_options

def fresh_rpc_path(root):
    """Atomic unique attempt directory; retained after shutdown for traceability."""
    return Path(tempfile.mkdtemp(prefix='w-',dir=root))

def _worker(connection,spec):
    # Spawn a fresh interpreter; establish device and IEEE mode before imports.
    os.setsid()  # Dedicated group: recovery can reap only this worker and its children.
    os.environ['CUDA_VISIBLE_DEVICES']=spec['cuda_mask']
    numeric_mode=spec.get('numeric_mode','ieee_fp32')
    os.environ['TRITON_F32_DEFAULT']='tf32' if numeric_mode=='tf32_pilot' else 'ieee'
    os.environ['VLLM_RPC_BASE_PATH']=spec['rpc_path']
    from transformers import AutoTokenizer
    from .vllm_two_stage import VLLMEngine
    from .asset_identity import local_model_identity
    from .audit import attempt
    engine=None
    try:
        if not Path(spec['rpc_path']).is_dir():raise ValueError('missing dedicated RPC directory')
        with attempt(Path(spec['evidence'])):
            identity=local_model_identity(spec['model'])
            if identity!=spec['assets']:raise ValueError('published sampling snapshot changed')
            tok=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True,trust_remote_code=False)
            engine=VLLMEngine(spec['model'],tok,spec['policy'],max_context_tokens=spec['context'],
                              runtime_options=spec['runtime_options'],numeric_mode=numeric_mode)
            if local_model_identity(spec['model'])!=identity:raise ValueError('sampling snapshot changed during engine load')
            connection.send(('ready',dict(policy=spec['policy'],assets=identity,pid=os.getpid(),
                                          runtime_options=engine.runtime_options,numeric_mode=numeric_mode)))
            while True:
                command=connection.recv()
                if command[0]=='close':break
                _,requests,stage,policy=command
                connection.send(('result',engine.sample(requests,stage,policy)))
    except BaseException as exc:
        import traceback
        connection.send(('error',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
        raise
    finally:
        try:
            if engine is not None:engine.llm.llm_engine.engine_core.shutdown()
        finally:connection.close()

class WorkerTransport:
    def __init__(self,spec):
        ctx=multiprocessing.get_context('spawn');self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=_worker,args=(child,spec));self.closed=False
        self.process.start();child.close()
        try:
            kind,self.ready=self.receive()
            if kind!='ready':raise RuntimeError('worker did not acknowledge immutable policy')
        except BaseException:
            self._abort()
            raise
    def _abort(self):
        """Reap our own process group on timeout/error; never retry a request."""
        if self.closed:return
        try:
            if self.process.is_alive():
                try:
                    if os.getpgid(self.process.pid)==self.process.pid:os.killpg(self.process.pid,signal.SIGTERM)
                    else:self.process.terminate()
                except ProcessLookupError:pass
            self.process.join(5)
            if self.process.is_alive():
                try:
                    if os.getpgid(self.process.pid)==self.process.pid:os.killpg(self.process.pid,signal.SIGKILL)
                    else:self.process.kill()
                except ProcessLookupError:pass
                self.process.join(5)
            event('vllm_worker_aborted',exit_code=self.process.exitcode,request_retried=False)
        finally:
            self.closed=True;self.connection.close()
    def receive(self):
        if not self.connection.poll(1800):raise TimeoutError('vLLM worker response missing; no request retry')
        kind,value=self.connection.recv()
        if kind=='error':raise RuntimeError('vLLM worker failure: '+str(value))
        return kind,value
    def sample(self,requests,stage,policy):
        try:
            self.connection.send(('sample',requests,stage,policy));kind,result=self.receive()
            if kind!='result':raise RuntimeError('worker response mismatch')
            return result
        except BaseException:
            self._abort()
            raise
    def close(self):
        if self.closed:return
        try:
            self.connection.send(('close',));self.process.join(120)
            if self.process.is_alive() or self.process.exitcode!=0:raise RuntimeError('vLLM worker did not shut down cleanly')
        except BaseException:
            self._abort()
            raise
        self.closed=True;self.connection.close()

class PhaseSeparatedGenerator:
    """HF model/optimizer residency changes only; objective and histories stay fixed."""
    def __init__(self,*,actor,reference,optimizer,tokenizer,version,behavior,device,root,rpc_root,
                 max_new_tokens=4096,max_context_tokens=32768,vllm_runtime=None,numeric_mode='ieee_fp32'):
        self.actor,self.reference,self.optimizer=actor,reference,optimizer
        self.tokenizer,self.version,self.behavior=tokenizer,version,behavior
        self.device=device;self.root=Path(root);self.rpc_root=Path(rpc_root)
        self.root.mkdir(parents=True,exist_ok=False);self.rpc_root.mkdir(parents=True,exist_ok=True)
        masks=os.environ['CUDA_VISIBLE_DEVICES'].split(',');self.cuda_mask=masks[device.index]
        self.worker=None;self.sampler=None;self.serial=0;self.snapshots={};self.moves=[]
        self.max_new_tokens=max_new_tokens;self.max_context_tokens=max_context_tokens
        self.numeric_mode=numeric_mode
        self.vllm_runtime=vllm_runtime_options(vllm_runtime,numeric_mode=numeric_mode)
    @property
    def policy_version(self):return self.version()
    def _offload(self):
        import torch
        # Keep exact tensor values and original per-state devices (Adam step can be CPU).
        self.actor.to('cpu');self.reference.to('cpu');self.moves=[]
        for state in self.optimizer.state.values():
            for key,tensor in state.items():
                if isinstance(tensor,torch.Tensor) and tensor.device.type=='cuda':
                    self.moves.append((state,key,tensor.device));state[key]=tensor.to('cpu')
        torch.cuda.synchronize(self.device);torch.cuda.empty_cache()
    def prepare_generation(self):
        if self.worker is not None:
            if self.worker.ready['policy']!=self.policy_version:raise RuntimeError('stale vLLM process after weight update')
            return
        from .policy_identity import identify
        from .asset_identity import local_model_identity
        from transformers import AutoModelForCausalLM
        import torch
        start=time.monotonic();version=self.policy_version
        self._offload()
        actual,evidence=identify(self.actor,self.behavior)
        if actual!=version:raise ValueError('live Actor differs from registered behavior policy')
        if version not in self.snapshots:
            folder=self.root/('snapshot-'+version);folder.mkdir()
            self.actor.save_pretrained(folder/'actor',safe_serialization=True);self.tokenizer.save_pretrained(folder/'actor')
            reloaded=AutoModelForCausalLM.from_pretrained(folder/'actor',torch_dtype=torch.float32,local_files_only=True,trust_remote_code=False).eval()
            if identify(reloaded,self.behavior)[0]!=version:raise ValueError('sampling snapshot differs from live Actor')
            del reloaded
            assets=local_model_identity(folder/'actor')
            atomic_json(folder/'complete.json',dict(policy=version,assets=assets,state_dict_sha256=evidence['state_dict_sha256']))
            self.snapshots[version]=(folder/'actor',assets)
        model,assets=self.snapshots[version];self.serial+=1
        spec=dict(model=str(model),assets=assets,policy=version,cuda_mask=self.cuda_mask,context=self.max_context_tokens,
            runtime_options=self.vllm_runtime,numeric_mode=self.numeric_mode,
            rpc_path=str(fresh_rpc_path(self.rpc_root)),evidence=str(self.root/('worker-'+str(self.serial))))
        self.worker=WorkerTransport(spec)
        self.sampler=VLLMTwoStage(self.worker,self.tokenizer,self.version,self.max_new_tokens,self.max_context_tokens,
            backend=BACKEND if self.numeric_mode=='ieee_fp32' else 'vllm_mixed_precision_pilot_v1')
        atomic_json(self.root/('worker-'+str(self.serial)+'-ready.json'),dict(policy=version,assets=assets,
            runtime_options=self.vllm_runtime,worker_ready=self.worker.ready,load_seconds=time.monotonic()-start,
            cache_policy='new process; no KV reuse across model snapshots'))
        event('vllm_policy_ready',policy_version=version,runtime_options=self.vllm_runtime,
              load_seconds=time.monotonic()-start)
    def prepare_scoring(self):
        if self.worker is not None:
            self.worker.close();self.worker=None;self.sampler=None
        self.actor.to(self.device);self.reference.to(self.device)
        for state,key,device in self.moves:state[key]=state[key].to(device)
        self.moves=[]
    def generate_many(self,prefixes,seeds):
        self.prepare_generation();return self.sampler.generate_many(prefixes,seeds)
    def generate(self,prefix,*,seed):return self.generate_many([prefix],[seed])[0]
    def close(self):
        if self.worker is not None:self.worker.close();self.worker=None

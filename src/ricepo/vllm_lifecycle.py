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
from .audit import event,fingerprint,operation
from .artifacts import atomic_json
from .vllm_two_stage import VLLMTwoStage,BACKEND,vllm_runtime_options

def fresh_rpc_path(root):
    """Atomic unique attempt directory; retained after shutdown for traceability."""
    return Path(tempfile.mkdtemp(prefix='w-',dir=root))


def release_residency_cache(device):
    """Run after _offload returns, so its last CUDA tensor local is dead.

    Emptying the allocator inside that frame can leave its entire segment
    reserved until this second boundary. No live model/state tensor is dropped.
    """
    import gc,torch
    gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                free_device_bytes=free,total_device_bytes=total)

def isolate_hf_allocator(enabled):
    if enabled:
        for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):os.environ.pop(key,None)


def _worker(connection,spec):
    # Spawn a fresh interpreter; establish device and IEEE mode before imports.
    os.setsid()  # Dedicated group: recovery can reap only this worker and its children.
    # The HF capacity repair must not change the frozen sampling allocator.
    isolate_hf_allocator(spec.get('isolate_training_allocator'))
    os.environ['CUDA_VISIBLE_DEVICES']=spec['cuda_mask']
    if spec.get('cache_root'):
        from .environment import configure_caches
        configure_caches(spec['cache_root'])
    numeric_mode=spec.get('numeric_mode','ieee_fp32')
    os.environ['TRITON_F32_DEFAULT']='tf32' if numeric_mode in ('tf32_pilot','tf32_gemm_pilot','tf32_gemm_control') else 'ieee'
    if numeric_mode=='tf32_gemm_pilot':
        os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='high'
    if numeric_mode=='tf32_gemm_control':os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='highest'
    os.environ['VLLM_RPC_BASE_PATH']=spec['rpc_path']
    from transformers import AutoTokenizer
    from .vllm_two_stage import VLLMEngine
    from .asset_identity import local_model_identity
    from .audit import attempt
    engine=None
    try:
        if not Path(spec['rpc_path']).is_dir():raise ValueError('missing dedicated RPC directory')
        with attempt(Path(spec['evidence'])):
            if spec.get('reuse_verified_assets'):
                from .asset_identity import VerifiedModelIdentity
                identities=VerifiedModelIdentity();model_identity=identities.get
            else:model_identity=local_model_identity
            identity=model_identity(spec['model'])
            if identity!=spec['assets']:raise ValueError('published sampling snapshot changed')
            tok=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True,trust_remote_code=False)
            inspection={'worker_extension_cls':'ricepo.gemm_diagnostic.GemmInspection'} if numeric_mode in ('tf32_gemm_pilot','tf32_gemm_control') else {}
            engine=VLLMEngine(spec['model'],tok,spec['policy'],max_context_tokens=spec['context'],
                              runtime_options=spec['runtime_options'],numeric_mode=numeric_mode,co_resident=spec.get('co_resident',False),**inspection)
            gemm_state=engine.llm.collective_rpc('gemm_state') if inspection else None
            expected='high' if numeric_mode=='tf32_gemm_pilot' else 'highest'
            if inspection and not all(s['matmul_allow_tf32']==(expected=='high') and s['torch_float32_matmul_precision']==expected for s in gemm_state):
                raise ValueError('actual GEMM worker precision differs from frozen candidate')
            if spec.get('isolate_training_allocator') and (not gemm_state or any(s.get('allocator_expandable_segments') is not False for s in gemm_state)):
                raise ValueError('HF allocator repair leaked into the sampling worker')
            if model_identity(spec['model'])!=identity:raise ValueError('sampling snapshot changed during engine load')
            connection.send(('ready',dict(policy=spec['policy'],assets=identity,pid=os.getpid(),
                runtime_options=engine.runtime_options,numeric_mode=numeric_mode,
                **({'gemm_state':gemm_state} if inspection else {}))))
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
    def __init__(self,spec,*,wait=True):
        response_timeout_seconds=spec.get('response_timeout_seconds',1800)
        if type(response_timeout_seconds) is not int or not 0<response_timeout_seconds<=172800:
            raise ValueError('invalid bounded worker response wait')
        self.response_timeout_seconds=1800  # initialization still has the existing short failure bound
        ctx=multiprocessing.get_context('spawn');self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=_worker,args=(child,spec));self.closed=False
        self.process.start();child.close()
        self.steady_timeout=response_timeout_seconds
        if wait:self.wait_ready()
    def wait_ready(self):
        try:
            kind,self.ready=self.receive()
            if kind!='ready':raise RuntimeError('worker did not acknowledge immutable policy')
            self.response_timeout_seconds=self.steady_timeout
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
        if not self.connection.poll(getattr(self,'response_timeout_seconds',1800)):raise TimeoutError('vLLM worker response missing; no request retry')
        kind,value=self.connection.recv()
        if kind=='error':raise RuntimeError('vLLM worker failure: '+str(value))
        return kind,value
    def start_sample(self,requests,stage,policy):
        self.connection.send(('sample',requests,stage,policy))
    def finish_sample(self):
        kind,result=self.receive()
        if kind!='result':raise RuntimeError('worker response mismatch')
        return result
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
                 max_new_tokens=4096,max_context_tokens=32768,vllm_runtime=None,numeric_mode='ieee_fp32',
                 reuse_verified_assets=False,validated_policy_getter=None,response_timeout_seconds=1800,
                 defer_optimizer_residency=False,isolate_training_allocator=False,
                 dual_runtime=None,cache_root=None,phase_callback=None):
        if type(defer_optimizer_residency) is not bool or type(isolate_training_allocator) is not bool:
            raise ValueError('residency controls must be explicit booleans')
        self.actor,self.reference,self.optimizer=actor,reference,optimizer
        self.tokenizer,self.version,self.behavior=tokenizer,version,behavior
        self.device=device;self.root=Path(root);self.rpc_root=Path(rpc_root)
        self.root.mkdir(parents=True,exist_ok=False);self.rpc_root.mkdir(parents=True,exist_ok=True)
        masks=os.environ['CUDA_VISIBLE_DEVICES'].split(',');self.cuda_mask=masks[device.index]
        self.worker=None;self.sampler=None;self.serial=0;self.snapshots={};self.moves=[]
        self.phase_callback=phase_callback
        self.cache_root=Path(cache_root) if cache_root else None
        self.secondary_mask=masks[1] if dual_runtime is not None else None
        self.dual_runtime=vllm_runtime_options(dual_runtime,numeric_mode=numeric_mode,allow_low_memory=True) if dual_runtime is not None else None
        self.affinity={}
        self.current_group_keys=None
        self.max_new_tokens=max_new_tokens;self.max_context_tokens=max_context_tokens
        self.numeric_mode=numeric_mode
        self.reuse_verified_assets=reuse_verified_assets;self.validated_policy_getter=validated_policy_getter
        self.response_timeout_seconds=response_timeout_seconds
        self.defer_optimizer_residency=defer_optimizer_residency
        self.isolate_training_allocator=isolate_training_allocator
        self.vllm_runtime=vllm_runtime_options(vllm_runtime,numeric_mode=numeric_mode)
    @property
    def policy_version(self):return self.version()
    def _offload(self):
        import torch
        # Keep exact tensor values and original per-state devices (Adam step can be CPU).
        self.actor.to('cpu');self.reference.to('cpu')
        if not getattr(self,'defer_optimizer_residency',False):self.moves=[]
        for state in self.optimizer.state.values():
            for key,tensor in state.items():
                if isinstance(tensor,torch.Tensor) and tensor.device.type=='cuda':
                    self.moves.append((state,key,tensor.device));state[key]=tensor.to('cpu')
        torch.cuda.synchronize(self.device);torch.cuda.empty_cache()
    def prepare_generation(self):
        if self.worker is not None:
            if self.worker.ready["policy"]!=self.policy_version:raise RuntimeError("stale vLLM process after weight update")
            return
        with operation("model_lifecycle",transition="prepare_generation"):
            return self._prepare_generation()
    def _prepare_generation(self):
        if self.worker is not None:
            if self.worker.ready['policy']!=self.policy_version:raise RuntimeError('stale vLLM process after weight update')
            return
        from .policy_identity import identify
        from .asset_identity import local_model_identity
        from transformers import AutoModelForCausalLM
        import torch
        start=time.monotonic();version=self.policy_version
        self._offload()
        if self.phase_callback:self.phase_callback('generation')
        released=release_residency_cache(self.device)
        event('generation_residency_released',policy_version=version,**released)
        if self.actor.training:raise ValueError('behavior identity requires eval mode')
        from .asset_identity import tensor_versions
        verified=self.validated_policy_getter() if self.validated_policy_getter else None
        if (verified and verified.get("policy")==version and verified.get("tensor_versions")==tensor_versions(self.actor)):
            actual,evidence=version,verified["evidence"]
        else:actual,evidence=identify(self.actor,self.behavior)
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
            isolate_training_allocator=self.isolate_training_allocator,
            runtime_options=self.vllm_runtime,numeric_mode=self.numeric_mode,reuse_verified_assets=self.reuse_verified_assets,
            response_timeout_seconds=self.response_timeout_seconds,
            rpc_path=str(fresh_rpc_path(self.rpc_root)),evidence=str(self.root/('worker-'+str(self.serial))))
        if self.cache_root:spec['cache_root']=str(self.cache_root/('primary-'+str(self.serial)))
        if self.dual_runtime is None:self.worker=WorkerTransport(spec)
        else:
            secondary={**spec,'cuda_mask':self.secondary_mask,'runtime_options':self.dual_runtime,'co_resident':True,
                'rpc_path':str(fresh_rpc_path(self.rpc_root)),
                'evidence':str(self.root/('secondary-'+str(self.serial)))}
            if self.cache_root:secondary['cache_root']=str(self.cache_root/('secondary-'+str(self.serial)))
            self.worker=DualTransport([spec,secondary],self.affinity,self)
        self.sampler=VLLMTwoStage(self.worker,self.tokenizer,self.version,self.max_new_tokens,self.max_context_tokens,
            backend=BACKEND if self.numeric_mode=='ieee_fp32' else 'vllm_mixed_precision_pilot_v1')
        atomic_json(self.root/('worker-'+str(self.serial)+'-ready.json'),dict(policy=version,assets=assets,
            runtime_options=self.vllm_runtime,worker_ready=self.worker.ready,load_seconds=time.monotonic()-start,
            cache_policy='new process; no KV reuse across model snapshots'))
        event('vllm_policy_ready',policy_version=version,runtime_options=self.vllm_runtime,
              load_seconds=time.monotonic()-start)
    def prepare_scoring(self):
        with operation("model_lifecycle",transition="prepare_scoring"):
            return self._prepare_scoring()
    def _prepare_scoring(self):
        if self.worker is not None:
            self.worker.close();self.worker=None;self.sampler=None
        self.actor.to(self.device);self.reference.to(self.device)
        if self.phase_callback:self.phase_callback('scoring')
        if getattr(self,'defer_optimizer_residency',False):return
        self.prepare_optimizer_step()
    def prepare_optimizer_step(self):
        """Moments are unused by forward/backward; restore exact values for Adam."""
        if self.worker is not None:raise RuntimeError('cannot restore optimizer while sampling is active')
        if self.phase_callback:self.phase_callback('optimizer')
        for state,key,device in self.moves:state[key]=state[key].to(device)
        self.moves=[]
    def generate_many(self,prefixes,seeds):
        self.prepare_generation();return self.sampler.generate_many(prefixes,seeds)
    def generate_many_with_identities(self,prefixes,seeds,identities):
        if len(identities)!=len(prefixes):raise ValueError('missing scheduling identities')
        self.current_group_keys=[affinity_key(i) for i in identities]
        try:return self.generate_many(prefixes,seeds)
        finally:self.current_group_keys=None
    def generate(self,prefix,*,seed):return self.generate_many([prefix],[seed])[0]
    def close(self):
        if self.worker is not None:self.worker.close();self.worker=None


def affinity_key(identity):
    """All originals of a query or all alternatives of an anchor stay together."""
    import re
    t=identity['trajectory_id']
    match=re.fullmatch(r'(.*-d[0-9]+)-b[0-9]+',t)
    return 'anchor:'+match.group(1) if match else 'query:'+identity['query_id']


def affine_partitions(requests,keys,affinity,workers=2):
    if len(requests)!=len(keys):raise ValueError('misaligned affinity identities')
    groups={}
    for i,key in enumerate(keys):groups.setdefault(key,[]).append(i)
    loads=[0]*workers;bins=[[] for _ in range(workers)]
    # Fixed prior ownership is preserved; new groups use full native input+budget cost.
    costs={key:sum(len(requests[i]['prefix'])+requests[i]['budget'] for i in ids) for key,ids in groups.items()}
    for key,ids in groups.items():
        if key in affinity:
            worker=affinity[key];bins[worker].extend(ids);loads[worker]+=costs[key]
    for key in sorted((k for k in groups if k not in affinity),key=lambda k:(-costs[k],k)):
        worker=min(range(workers),key=lambda i:(loads[i],i));affinity[key]=worker
        bins[worker].extend(groups[key]);loads[worker]+=costs[key]
    if sorted(i for b in bins for i in b)!=list(range(len(requests))):raise ValueError('scheduler lost/duplicated request')
    return bins


class DualTransport:
    def __init__(self,specs,affinity,owner):
        self.workers=[];self.affinity=affinity;self.owner=owner
        try:
            for spec in specs:
                self.workers.append(WorkerTransport(spec,wait=False))
            for w in self.workers:w.wait_ready()
            first=self.workers[0].ready
            if any(w.ready['policy']!=first['policy'] or w.ready['assets']!=first['assets'] for w in self.workers):
                raise ValueError('dual engines differ in immutable weight identity')
            self.ready={**first,'workers':[w.ready for w in self.workers]}
        except BaseException:
            for w in self.workers:w._abort()
            raise
    def sample(self,requests,stage,policy):
        keys=[self.owner.current_group_keys[r['request_slot']] if self.owner.current_group_keys is not None
              else 'prefix:'+fingerprint(r['prefix']) for r in requests]
        bins=affine_partitions(requests,keys,self.affinity)
        results=[None]*len(requests)
        try:
            for w,ids in zip(self.workers,bins,strict=True):
                if ids:w.start_sample([requests[i] for i in ids],stage,policy)
            for w,ids in zip(self.workers,bins,strict=True):
                if not ids:continue
                values=w.finish_sample()
                if len(values)!=len(ids):raise ValueError('dual worker lost request')
                for i,value in zip(ids,values,strict=True):results[i]=value
            if any(r is None for r in results):raise ValueError('dual response incomplete')
            event('dual_engine_dispatch',stage_name=stage,policy_version=policy,
                request_counts=[len(x) for x in bins],group_keys_sha256=fingerprint(keys),
                input_tokens=[sum(len(requests[i]['prefix']) for i in ids) for ids in bins])
            return results
        except BaseException:
            for w in self.workers:w._abort()
            raise
    def close(self):
        errors=[]
        for w in self.workers:
            try:w.close()
            except BaseException as exc:errors.append(exc)
        if errors:raise errors[0]

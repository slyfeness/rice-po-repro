"""Spawned, fixed Retriever owner; JVM and Diver never live in the HF parent."""
import multiprocessing
import os
import signal
from pathlib import Path


def _serve(connection,spec):
    os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES']=spec['cuda_mask']
    from .environment import configure_caches
    configure_caches(spec['cache_root'])
    from .audit import attempt
    try:
        with attempt(Path(spec['cache_root'])/'evidence'):
            import torch
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            from transformers import AutoTokenizer
            from .formats import PromptRenderer
            from .retrieval import DiverEncoder
            from .hf_runtime import _domain_retriever
            dep={**spec['deployment'],'retriever_device':'cuda:0','cache_root':spec['cache_root']}
            plan=spec['plan']
            tok=AutoTokenizer.from_pretrained(dep['actor_model_path'],local_files_only=True,trust_remote_code=False)
            renderer=PromptRenderer(tok,Path(spec['prompt_path']).read_text('utf8'),
                observation_top_k=plan['observation_top_k'],observation_token_limit=plan['observation_token_limit'],prefill_think=True)
            encoder=DiverEncoder.from_local(dep)
            backends={};sources={}
            for domain,domain_spec in dep['domains'].items():
                backends[domain],sources[domain]=_domain_retriever(domain,domain_spec,dep,plan,
                    spec['model_spec'],encoder,renderer,spec['examples'],spec['qrels'])
            connection.send(('ready',dict(sources=sources,encoder_identity=encoder.identity)))
            while True:
                command=connection.recv()
                if command[0]=='close':break
                if command[0]=='residency':
                    import time
                    phase=command[1]
                    if phase not in ('generation','hf'):raise ValueError('unknown retriever residency phase')
                    started=time.monotonic()
                    if phase=='hf':
                        encoder.model.to('cpu')
                        torch.cuda.synchronize();torch.cuda.empty_cache()
                        device='cpu'
                    else:
                        encoder.model.to(encoder.device)
                        torch.cuda.synchronize()
                        device=str(encoder.device)
                    connection.send(('residency',dict(phase=phase,device=device,seconds=time.monotonic()-started)))
                    continue
                _,domain,requests=command
                if next(encoder.model.parameters()).device.type!='cuda':
                    raise RuntimeError('retrieval requested while Diver is offloaded for HF')
                values=backends[domain].retrieve_many(requests)
                connection.send(('result',values))
    except BaseException as exc:
        import traceback
        connection.send(('error',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
        raise
    finally:connection.close()


class _Domain:
    def __init__(self,owner,domain,backend_version):
        self.owner,self.domain,self.backend_version=owner,domain,backend_version
    def retrieve(self,query,*,excluded_ids):
        return self.retrieve_many([dict(query=query,excluded_ids=excluded_ids)])[0]
    def retrieve_many(self,requests):
        self.owner.connection.send(('retrieve',self.domain,requests))
        kind,values=self.owner.receive()
        if kind!='result' or len(values)!=len(requests):raise RuntimeError('retrieval response identity/count mismatch')
        return values


class RemoteRetrievalService:
    def __init__(self,deployment,plan,model_spec,examples,qrels,prompt_path):
        index=int(deployment['retriever_device'].split(':')[-1])
        spec=dict(deployment=deployment,plan=plan,model_spec=model_spec,examples=examples,qrels=qrels,
            prompt_path=str(prompt_path),cache_root=str(Path(deployment['cache_root'])/'retrieval-process'),
            cuda_mask=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[index])
        ctx=multiprocessing.get_context('spawn');self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=_serve,args=(child,spec));self.closed=False;self.transitions=[]
        self.process.start();child.close()
        try:
            kind,ready=self.receive()
            if kind!='ready':raise RuntimeError('retriever missing initialization identity')
            self.sources=ready['sources'];self.encoder_identity=ready['encoder_identity']
            self.retrievers={d:_Domain(self,d,s['backend_version']) for d,s in self.sources.items()}
        except BaseException:self.close();raise
    def receive(self):
        if not self.connection.poll(7200):raise TimeoutError('retriever process response missing; no retry')
        kind,value=self.connection.recv()
        if kind=='error':raise RuntimeError('retriever process failed: '+str(value))
        return kind,value
    def set_residency(self,phase):
        if phase not in ('generation','hf'):raise ValueError('unknown retriever residency phase')
        self.connection.send(('residency',phase))
        kind,value=self.receive()
        if kind!='residency' or value.get('phase')!=phase:
            raise RuntimeError('retriever residency transition did not acknowledge the requested phase')
        if (phase=='generation') != (value.get('device')!='cpu'):
            raise RuntimeError('retriever residency acknowledgement has the wrong device')
        self.transitions.append(value)
        return value
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.process.is_alive():
                self.connection.send(('close',));self.process.join(15)
                if self.process.is_alive():
                    if os.getpgid(self.process.pid)==self.process.pid:os.killpg(self.process.pid,signal.SIGTERM)
                    else:self.process.terminate()
                    self.process.join(10)
        finally:self.connection.close()

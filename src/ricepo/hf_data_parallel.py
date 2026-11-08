"""Persistent two-rank FP32 HF replica. Imports never initialize CUDA."""
from contextlib import contextmanager
from pathlib import Path
import datetime
import time


def validate_spec(spec):
    if spec != dict(world_size=2, partition='length_greedy_v1', grad_reduce='manual_allreduce_sum'):
        raise ValueError('HF DP requires two length-greedy ranks and manual SUM gradients')


def validate_recipe(recipe, global_size=None):
    if recipe['ppo_epochs']!=1 or recipe['loss_reduction']!='sequence_mean' or recipe['max_grad_norm']!=1:
        raise ValueError('HF DP requires one full-batch sequence-mean update with clip norm 1')
    for key in ('optimizer_minibatch_size','gradient_accumulation'):
        value=recipe[key]
        if value=='full_batch':continue
        if type(value) is not int or value<1 or (global_size is not None and value!=global_size):
            raise ValueError('HF DP minibatch and accumulation must equal the global batch size')


def length_partition(items):
    shards, loads = [[], []], [0, 0]
    def horizon(i):
        return max(j for j, flag in enumerate(items[i].generated_mask) if flag)
    for i in sorted(range(len(items)), key=lambda i: (-horizon(i), i)):
        rank = min(range(2), key=lambda r: (loads[r], r))
        shards[rank].append(i); loads[rank] += horizon(i)
    return shards


@contextmanager
def hf_precision(mode):
    import torch
    old=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision('high' if mode=='tf32' else 'highest')
    torch.backends.cuda.matmul.allow_tf32=mode=='tf32'; torch.backends.cudnn.allow_tf32=False
    try: yield
    finally:
        torch.set_float32_matmul_precision(old[0]);torch.backends.cuda.matmul.allow_tf32=old[1];torch.backends.cudnn.allow_tf32=old[2]


def _move_optimizer(optimizer, device):
    # Adam's scalar step stays on CPU for the standard noncapturable optimizer.
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, 'to') and key != 'step':state[key]=value.to(device)


class Replica:
    def __init__(self, actor, reference, optimizer, device, options, recipe, behavior, precision):
        self.actor,self.reference,self.optimizer,self.device=actor,reference,optimizer,device
        self.options,self.recipe,self.behavior,self.precision=options,recipe,behavior,precision
        self.stage_seconds={}
    def phase(self, phase):
        import torch
        if phase=='generation':
            self.optimizer.zero_grad(set_to_none=True)
            self.actor.to('cpu');self.reference.to('cpu');_move_optimizer(self.optimizer,'cpu')
            torch.cuda.empty_cache()
        elif phase=='scoring':
            self.actor.to(self.device);self.reference.to(self.device)
        elif phase=='optimizer':_move_optimizer(self.optimizer,self.device)
        else:raise ValueError('unknown HF phase')
    def rng(self, value=None):
        import torch
        if value is not None:
            torch.random.set_rng_state(value['cpu_rng']);torch.cuda.set_rng_state(value['cuda_rng'],self.device)
        return dict(cpu_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.device))
    def score(self, trajectories, entropy=False):
        import torch
        from .backends import TransformersEntropyProvider
        from .modeling import sampled_old_log_probs
        result={}
        # Entropy belongs to the frozen behavior policy for this iteration.
        # Do not advertise a placeholder identity: the strict provider must
        # receive the actual native trajectory policy version.
        versions={t.policy_version for t in trajectories}
        if len(versions)!=1 or not next(iter(versions),None):
            raise ValueError('DP scoring requires one nonempty frozen policy version')
        policy_version=next(iter(versions))
        with hf_precision('ieee'),torch.no_grad():
            provider=TransformersEntropyProvider(model=self.actor,policy_version_getter=lambda:policy_version,device=self.device,
                chunk_size=self.options['logit_chunk_size'],max_context_tokens=self.options['max_context_tokens'],model_call_kwargs={})
            for t in trajectories:
                result[t.trajectory_id]=(tuple(provider.token_entropies(t)) if entropy else tuple(sampled_old_log_probs(self.actor,t.token_ids,t.generated_mask,
                    device=self.device,logit_chunk_size=self.options['logit_chunk_size'],max_context_tokens=self.options['max_context_tokens'],policy_call_kwargs={},
                    projection_mode=self.options['projection_mode'])[0].cpu().tolist()))
        return result
    def backward(self, items, indices):
        import torch
        from .modeling import make_forward
        from .objective import ppo_loss
        torch.cuda.synchronize(self.device);started=time.monotonic()
        recipe=self.recipe;self.optimizer.zero_grad(set_to_none=True)
        forward=make_forward(reference_model=self.reference,device=self.device,policy_call_kwargs={},reference_call_kwargs={},**self.options)
        rows=[]
        with hf_precision(self.precision):
            for i in indices:
                item=items[i];current,kl=forward(self.actor,item)
                old=torch.tensor([item.old_log_probs],device=self.device,dtype=current.dtype)
                adv=torch.tensor([item.advantages],device=self.device,dtype=current.dtype)
                mask=torch.tensor([item.generated_mask],device=self.device,dtype=torch.bool)
                loss=ppo_loss(current,old,adv,mask,clip_epsilon=recipe['clip_epsilon'],kl_coefficient=recipe['actor_kl_coefficient'],reference_kl=kl,reduction='sum')
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite DP loss')
                (loss/len(items)).backward()
                rows.append((i,float(loss.detach()),current.detach().cpu(),float(kl.detach()[mask].double().sum())))
        torch.cuda.synchronize(self.device);self.stage_seconds['forward_backward']=time.monotonic()-started
        return rows
    def timings(self):return dict(self.stage_seconds)
    def clear_gradients(self):
        self.optimizer.zero_grad(set_to_none=True)
    def reduce(self):
        import torch
        import torch.distributed as dist
        torch.cuda.synchronize(self.device);started=time.monotonic()
        flags=[p.grad is not None for p in self.actor.parameters()];both=[None,None]
        dist.all_gather_object(both,flags)
        if both[0]!=both[1]:raise ValueError('rank gradient presence differs')
        for p in self.actor.parameters():
            if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device);self.stage_seconds['gradient_sum']=time.monotonic()-started
    def step(self):
        import torch
        from .policy_identity import identify
        torch.cuda.synchronize(self.device);started=time.monotonic()
        self.phase('optimizer')
        norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.recipe['max_grad_norm'],error_if_nonfinite=True)
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device);self.stage_seconds['clip_adam']=time.monotonic()-started
        started=time.monotonic()
        sha=identify(self.actor,self.behavior)[1]['state_dict_sha256']
        self.stage_seconds['parameter_sha']=time.monotonic()-started
        return float(norm),sha


def _worker(pipe, spec):
    import torch
    import torch.distributed as dist
    try:
        device=torch.device(spec['device']);torch.cuda.set_device(device)
        dist.init_process_group('nccl',rank=1,world_size=2,init_method=spec['init_method'],timeout=datetime.timedelta(seconds=120))
        from transformers import AutoModelForCausalLM
        from .modeling import enable_eval_layer_checkpointing
        def load(path):return AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,local_files_only=True).eval().to(device)
        actor,reference=load(spec['actor_path']),load(spec['reference_path'])
        reference.requires_grad_(False)
        enable_eval_layer_checkpointing(actor)
        hp=spec['recipe']['optimizer_hyperparameters']
        optimizer=torch.optim.AdamW(actor.parameters(),lr=spec['recipe']['actor_learning_rate'],betas=tuple(hp['betas']),eps=hp['eps'],weight_decay=hp['weight_decay'])
        from .checkpoint_io import state_digest
        canonical=(torch.load(spec['optimizer_path'],map_location='cpu',weights_only=True)['optimizer'] if 'optimizer_path' in spec else spec['optimizer'])
        if state_digest(canonical)!=spec['optimizer_sha256']:raise ValueError('canonical Adam payload digest differs')
        optimizer.load_state_dict(canonical)
        del canonical
        # Loading may move moment tensors to CUDA; verify values against the canonical CPU payload.
        if state_digest(optimizer.state_dict())!=spec['optimizer_sha256']:raise ValueError('rank1 restored Adam differs')
        replica=Replica(actor,reference,optimizer,device,spec['options'],spec['recipe'],spec['behavior'],spec['precision'])
        replica.rng(spec.get('rng'))
        from .policy_identity import identify
        pipe.send(('ok',(identify(actor,spec['behavior'])[1]['state_dict_sha256'],identify(reference,spec['behavior'])[1]['state_dict_sha256'])))
        while True:
            command,args=pipe.recv()
            if command=='close':break
            try:
                value=getattr(replica,command)(*args)
            except BaseException as exc:
                replica.clear_gradients()
                pipe.send(('error',f'{type(exc).__name__}: {exc}'))
                continue
            pipe.send(('ok',value))
    except BaseException as exc:
        pipe.send(('error',f'{type(exc).__name__}: {exc}'))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


class HFDataParallel:
    def __init__(self, *, actor,reference,optimizer,device,spec,worker_spec,rpc_root,precision,scoring_options,recipe,behavior):
        import torch
        import torch.multiprocessing as mp
        import torch.distributed as dist
        torch.cuda.set_device(device)
        from .policy_identity import identify
        validate_spec(spec)
        if precision not in ('ieee','tf32'):raise ValueError('HF precision must be ieee or tf32')
        validate_recipe(recipe)
        self.local=Replica(actor,reference,optimizer,device,scoring_options,recipe,behavior,precision)
        root=Path(rpc_root);root.mkdir(parents=True,exist_ok=True)
        import uuid
        init=(root/('hf-dp-'+uuid.uuid4().hex)).resolve().as_uri()
        context=mp.get_context('spawn');self.pipe,child=context.Pipe()
        worker_spec=dict(worker_spec,options=scoring_options,recipe=recipe,behavior=behavior,precision=precision,init_method=init)
        self.process=context.Process(target=_worker,args=(child,worker_spec));self.process.start();child.close()
        try:
            dist.init_process_group('nccl',rank=0,world_size=2,init_method=init,timeout=datetime.timedelta(seconds=120))
            actor_sha,reference_sha=self._receive()
            if actor_sha!=identify(actor,behavior)[1]['state_dict_sha256']:raise ValueError('initial rank Actor SHA differs')
            if reference_sha!=identify(reference,behavior)[1]['state_dict_sha256']:raise ValueError('initial rank Ref SHA differs')
        except BaseException:
            self.close();raise
    def _receive(self):
        deadline=time.monotonic()+43200
        while not self.pipe.poll(1):
            if not self.process.is_alive():raise RuntimeError(f'HF rank 1 exited with code {self.process.exitcode}')
            if time.monotonic()>=deadline:raise TimeoutError('HF rank 1 timed out')
        status,value=self.pipe.recv()
        if status!='ok':raise RuntimeError(value)
        return value
    def phase_callback(self, phase):
        self.pipe.send(('phase',(phase,)));self.local.phase(phase);self._receive()
    def prepare_generation(self):self.phase_callback('generation')
    def prepare_scoring(self):self.phase_callback('scoring')
    def prepare_optimizer_step(self):self.phase_callback('optimizer')
    def _scores(self, trajectories, entropy):
        trajectories=list(trajectories)
        # Entropy may contain terminal failures without actions; callers omit those.
        shards=length_partition(trajectories)
        self.pipe.send(('score',([trajectories[i] for i in shards[1]],entropy)))
        result=self.local.score([trajectories[i] for i in shards[0]],entropy);result.update(self._receive())
        if len(result)!=len(trajectories):raise ValueError('HF score coverage differs')
        return result
    def token_entropies_many(self, trajectories):return self._scores(trajectories,True)
    def score_many(self, trajectories):return self._scores(trajectories,False)
    def rng_state(self):
        self.pipe.send(('rng',()));return self._receive()
    def restore_rng_state(self,value):
        self.pipe.send(('rng',(value,)));self._receive()
    def _update_send(self,command,args=()):
        self.pipe.send((command,args));self._pending_update_rpc=True
    def _update_receive(self):
        try:return self._receive()
        finally:self._pending_update_rpc=False
    def _clear_failed_update(self):
        self.local.clear_gradients()
        if self._pending_update_rpc:
            # Drain the outstanding response before requesting the cleanup ACK.
            try:self._update_receive()
            except RuntimeError:pass
        self._update_send('clear_gradients')
        self._update_receive()
    def update(self,items,*,policy_version,probability_guard,step_observer=None,**unused):
        validate_recipe(self.local.recipe,len(items))
        self._pending_update_rpc=False
        try:
            return self._update_impl(items,policy_version=policy_version,probability_guard=probability_guard,step_observer=step_observer)
        except BaseException as exc:
            try:self._clear_failed_update()
            except BaseException as cleanup_error:
                exc.add_note(f'Both-rank gradient cleanup could not be acknowledged: {cleanup_error}')
            raise
    def _update_impl(self,items,*,policy_version,probability_guard,step_observer):
        import torch
        from .training import _validate_items
        from .policy_identity import identify
        _validate_items(items,policy_version)
        if probability_guard is None:raise ValueError('HF DP requires complete original risk gate')
        before=identify(self.local.actor,self.local.behavior)[1]['state_dict_sha256']
        shards=length_partition(items)
        self._update_send('backward',(items,shards[1]))
        rows=self.local.backward(items,shards[0])+self._update_receive();rows.sort(key=lambda r:r[0])
        if [r[0] for r in rows]!=list(range(len(items))):raise ValueError('DP update lost or duplicated items')
        probability_guard.begin(items,policy_version)
        for i,loss,current,kl in rows:probability_guard.observe(items[i],current)
        # The complete original risk and existing branch expansion gate precede either step.
        probability_guard.finish()
        self._update_send('reduce');self.local.reduce();self._update_receive()
        if step_observer:step_observer('before_clip')
        self._update_send('step');norm,sha=self.local.step();other_norm,other_sha=self._update_receive()
        if sha!=other_sha:raise ValueError('post-Adam rank Actor SHA differs; update cannot commit')
        if norm!=other_norm:raise ValueError('rank clipping norms differ')
        if step_observer:step_observer('after_step')
        self._update_send('timings');rank_timings=[self.local.timings(),self._update_receive()]
        report=probability_guard.report
        tokens=sum(sum(i.generated_mask) for i in items);kl=sum(row[3] for row in rows)
        loss=sum(row[1] for row in rows)/len(items)
        return dict(rank_stage_seconds=rank_timings,optimizer_steps=1,global_sequences=len(items),global_generated_tokens=tokens,normalized_loss=loss,
            policy_loss=loss-self.local.recipe['actor_kl_coefficient']*kl/len(items),reference_kl=kl/len(items),reference_kl_token_mean=kl/tokens,
            grad_norm_before_clip=norm,rank_weight_sha256=[sha,other_sha],rank_weight_hash_equal=True,actual_state_dict_changed=sha!=before,
            probability_statistics=report,parameter_probe_changed=sha!=before,zero_gradient_all_steps=norm==0,
            ratio_min=report['ratio']['min'],ratio_max=report['ratio']['max'],ratio_mean=report['ratio']['mean'],
            outside_clip_fraction=report['outside_clip_fraction'],effective_clip_fraction=report['effective_clip_fraction'])
    def close(self):
        import torch.distributed as dist
        if self.process.is_alive():
            try:self.pipe.send(('close',()))
            except (EOFError,BrokenPipeError,OSError):pass
            self.process.join(timeout=30)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=10)
        self.pipe.close()
        if dist.is_initialized():dist.destroy_process_group()

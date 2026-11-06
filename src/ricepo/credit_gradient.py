"""One bounded, same-originals RICE/GRPO gradient comparison, no baseline step."""
import math
from .objective import ppo_loss
from .credit import trajectory_advantages

def prepare_comparison(model,optimizer,originals,items,*,forward,device,recipe):
    import torch
    if recipe['ppo_epochs']!=1 or recipe['optimizer_minibatch_size']!=len(items):
        raise ValueError('credit gradient diagnostic requires the frozen single full-batch update')
    groups={}
    for t in originals:groups.setdefault(t.query_id,[]).append(t)
    baseline={}
    for group in groups.values():
        b=trajectory_advantages(group,epsilon=recipe['normalization_epsilon'])
        baseline.update(zip(b.trajectory_ids,b.advantages,strict=True))
    changed=sum(a!=b for item in items for a,b in zip(item.advantages,baseline[item.trajectory_id],strict=True))
    report=dict(changed_advantage_positions=changed,baseline_optimizer_steps=0)
    if not changed:
        report['status']='no_local_advantage_difference';return None,report
    def snapshot_grad():return {n:None if p.grad is None else p.grad.detach().to('cpu',copy=True) for n,p in model.named_parameters() if p.requires_grad}
    def squared_norm(rows):return math.fsum(float(v.double().square().sum()) for v in rows.values() if v is not None)
    denominator={'sum':1,'sequence_mean':len(items),'token_mean':sum(sum(i.generated_mask) for i in items)}[recipe['loss_reduction']]
    optimizer.zero_grad(set_to_none=True)
    for item in items:
        current,kl=forward(model,item)
        old=torch.tensor([item.old_log_probs],device=device,dtype=current.dtype)
        advantage=torch.tensor([baseline[item.trajectory_id]],device=device,dtype=current.dtype)
        mask=torch.tensor([item.generated_mask],device=device,dtype=torch.bool)
        loss=ppo_loss(current,old,advantage,mask,clip_epsilon=recipe['clip_epsilon'],
            kl_coefficient=recipe['actor_kl_coefficient'],reference_kl=kl,reduction='sum')
        (loss/denominator).backward()
        del current,kl,loss
    base=snapshot_grad();base_norm=squared_norm(base)
    report['grpo_gradient_norm']=math.sqrt(base_norm)
    optimizer.zero_grad(set_to_none=True)
    saved={}
    def observer(stage):
        nonlocal base
        if stage=='before_clip':
            grads=snapshot_grad();rice_norm=squared_norm(grads);delta_norm=dot=0.
            for name,g in grads.items():
                b=base[name]
                if g is None and b is None:continue
                if g is None:g=torch.zeros_like(b)
                if b is None:b=torch.zeros_like(g)
                delta_norm+=float((g.double()-b.double()).square().sum())
                dot+=float((g.double()*b.double()).sum())
            report.update(rice_gradient_norm=math.sqrt(rice_norm),gradient_difference_norm=math.sqrt(delta_norm),
                gradient_cosine=dot/math.sqrt(base_norm*rice_norm) if base_norm and rice_norm else None)
            base=None
            saved['grads']=grads
            saved['weights']={n:p.detach().to('cpu',copy=True) for n,p in model.named_parameters() if p.requires_grad}
        elif stage=='after_step':
            direction=delta_norm=0.
            for name,p in model.named_parameters():
                if not p.requires_grad:continue
                delta=p.detach().to('cpu',copy=True)-saved['weights'][name]
                g=saved['grads'][name]
                delta_norm+=float(delta.double().square().sum())
                if g is not None:direction+=float((g.double()*delta.double()).sum())
            report.update(status='compared_on_identical_originals',gradient_dot_parameter_delta=direction,
                parameter_delta_l2=math.sqrt(delta_norm),first_order_descent=direction<0)
            saved.clear()
        else:raise ValueError('unknown optimizer observation stage')
    return observer,report

"""Same originals/advantages/actual q; reference comparison has zero steps."""
import math
from .objective import ppo_loss


def prepare(model,optimizer,items,*,candidate_forward,ieee_forward,device,recipe,pilot):
    import torch
    if recipe['ppo_epochs']!=1 or recipe['optimizer_minibatch_size'] not in ('full_batch',len(items)):
        raise ValueError('pilot gradient comparison requires one full-batch step')
    denominator={'sum':1,'sequence_mean':len(items),'token_mean':sum(sum(i.generated_mask) for i in items)}[recipe['loss_reduction']]
    def gradients():
        return {n:p.grad.detach().to('cpu',copy=True) for n,p in model.named_parameters() if p.grad is not None}
    def norm(rows):return math.sqrt(math.fsum(float(x.double().square().sum()) for x in rows.values()))
    def backward(forward,coefficient):
        optimizer.zero_grad(set_to_none=True);total=0.
        for item in items:
            current,kl=forward(model,item)
            old=torch.tensor([item.old_log_probs],dtype=current.dtype,device=device)
            adv=torch.tensor([item.advantages],dtype=current.dtype,device=device)
            mask=torch.tensor([item.generated_mask],dtype=torch.bool,device=device)
            loss=ppo_loss(current,old,adv,mask,clip_epsilon=recipe['clip_epsilon'],kl_coefficient=coefficient,
                reference_kl=kl,reduction='sum')/denominator
            if not torch.isfinite(loss):raise FloatingPointError('pilot comparison loss nonfinite')
            loss.backward();total+=float(loss.detach())
            del current,kl,loss
        rows=gradients()
        if any(not bool(torch.isfinite(g).all()) for g in rows.values()):raise FloatingPointError('pilot comparison gradient nonfinite')
        return rows,total
    ieee,ieee_loss=backward(ieee_forward,recipe['actor_kl_coefficient'])
    policy,policy_loss=backward(candidate_forward,0.)
    optimizer.zero_grad(set_to_none=True)
    report=dict(ieee_loss=ieee_loss,candidate_policy_loss=policy_loss,ieee_gradient_norm=norm(ieee),
        candidate_policy_gradient_norm=norm(policy),reference_comparison_optimizer_steps=0,
        actual_q_and_advantages_fixed=True,actual_reward_advantage_positions=sum(a!=0 for i in items for a,m in zip(i.advantages,i.generated_mask) if m))
    state={}
    def observer(stage):
        nonlocal ieee,policy
        if stage=='before_clip':
            current=gradients();dot=diff=kl_norm=0.
            for name,g in current.items():
                r=ieee.get(name,torch.zeros_like(g));p=policy.get(name,torch.zeros_like(g))
                dot+=float((g.double()*r.double()).sum())
                diff+=float((g.double()-r.double()).square().sum())
                kl_norm+=float((g.double()-p.double()).square().sum())
            cn=norm(current);rn=report['ieee_gradient_norm']
            cosine=dot/(cn*rn) if cn and rn else None
            relative=math.sqrt(diff)/rn if rn else (0. if cn==0 else None)
            report.update(candidate_gradient_norm=cn,gradient_cosine=cosine,relative_gradient_l2=relative,
                candidate_kl_gradient_norm=math.sqrt(kl_norm))
            if cosine is not None and (cosine<pilot['gradient_screen'][pilot['candidate']+'_cosine_min']
                    or relative>pilot['gradient_screen'][pilot['candidate']+'_relative_l2_max']):
                raise ValueError('predeclared mixed-backend gradient risk screen failed; no optimizer step')
            state['before']={n:p.detach().to('cpu',copy=True) for n,p in model.named_parameters() if p.requires_grad}
            ieee=None
        elif stage=='after_step':
            squared=direction=0.;changed=0
            for name,p in model.named_parameters():
                if not p.requires_grad:continue
                d=p.detach().cpu()-state['before'][name]
                squared+=float(d.double().square().sum());changed+=int(torch.count_nonzero(d))
                if name in policy:direction+=float((d.double()*policy[name].double()).sum())
            report.update(parameter_delta_l2=math.sqrt(squared),changed_parameter_elements=changed,
                policy_gradient_dot_parameter_delta=direction,optimizer_step_executed=True,
                reward_gradient_nonzero=report['candidate_policy_gradient_norm']>0,
                fp32_master_parameters=all(p.dtype==torch.float32 for p in model.parameters()))
            state.clear();policy=None
        else:raise ValueError('unknown pilot optimizer event')
    return observer,report

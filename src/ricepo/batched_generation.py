"""Bounded HF two-stage batch sampler with one independent RNG per request.

This is an explicit unwarped (T=1, p=1, k=0) candidate. KV state exists only
inside each stage call; outputs contain immutable IDs, masks and scalar LPs.
No cache survives a model-weight change. Finished rows are never resampled.
"""
import inspect
import hashlib
import re
from .two_stage import MODE
from .audit import raw_completion,event,fingerprint
from .rollout import RolloutLengthError

def generate_many(owner,prefixes,seeds):
    import torch
    if owner.generation_protocol!=MODE or owner.model.training:raise ValueError('batch candidate requires frozen two-stage eval model')
    if len(prefixes)!=len(seeds) or not prefixes:raise ValueError('one seed per nonempty request required')
    prefixes=[tuple(p) for p in prefixes]
    if any(not p or any(type(t) is not int or t<0 for t in p) for p in prefixes):raise ValueError('invalid native batch prefix')
    if any(type(s) is not int or not 0<=s<2**63 for s in seeds):raise ValueError('invalid per-request seed')
    recipe=owner._generation_recipe(1)
    if (recipe.temperature,recipe.top_p,recipe.top_k,recipe.repetition_penalty)!=(1.,1.,0,1.):raise ValueError('unsupported batched sampling law')
    fmt=owner.two_stage_format;device=torch.device(owner.device);version=owner.policy_version
    generators=[torch.Generator(device=device).manual_seed(s) for s in seeds]
    pad=owner.tokenizer.pad_token_id
    if pad is None:pad=owner.tokenizer.eos_token_id
    eos=owner.tokenizer.eos_token_id
    kwargs={}
    signature=inspect.signature(owner.model.forward).parameters
    if 'logits_to_keep' in signature:kwargs['logits_to_keep']=1
    elif 'num_logits_to_keep' in signature:kwargs['num_logits_to_keep']=1
    def state_hash(g):return hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest()
    def stage(indices,histories,budgets,name,marker):
        n=len(indices);width=max(map(len,histories))
        inputs=torch.full((n,width),pad,dtype=torch.long,device=device)
        attention=torch.zeros_like(inputs)
        limits=[]
        for j,(history,budget) in enumerate(zip(histories,budgets,strict=True)):
            available=owner.max_context_tokens-len(history)
            if available<=0:raise RolloutLengthError('batch native context exhausted')
            limits.append(min(budget,available))
            inputs[j,-len(history):]=torch.tensor(history,dtype=torch.long,device=device)
            attention[j,-len(history):]=1
        positions=attention.cumsum(-1)-1;positions.masked_fill_(attention==0,0)
        out=owner.model(input_ids=inputs,attention_mask=attention,position_ids=positions,
            use_cache=True,return_dict=True,**kwargs)
        cache=out.past_key_values;logits=out.logits[:,-1,:]
        rows=[[] for _ in indices];lps=[[] for _ in indices];tails=[b'' for _ in indices]
        reasons=[None]*n;before=[state_hash(generators[i]) for i in indices]
        persisted=set()
        try:
            for step in range(max(limits)):
                if owner.policy_version!=version:raise RuntimeError('live policy changed during batch generation')
                tokens=torch.full((n,1),pad,dtype=torch.long,device=device)
                active=torch.zeros((n,1),dtype=torch.long,device=device)
                for j,i in enumerate(indices):
                    if reasons[j] is not None:continue
                    lp=torch.log_softmax(logits[j].float(),-1)
                    probs=torch.softmax(logits[j].float(),-1)
                    token=int(torch.multinomial(probs,1,generator=generators[i]))
                    rows[j].append(token);lps[j].append(float(lp[token]));tokens[j,0]=token;active[j,0]=1
                    raw=tails[j]+fmt.bytes.piece(token)
                    if marker in raw:reasons[j]='think_closed' if name=='think' else 'summary_closed'
                    elif token==eos:reasons[j]='eos'
                    elif len(rows[j])>=limits[j]:reasons[j]='length'
                    tails[j]=raw[-max(1,len(marker)-1):]
                    if reasons[j] is not None:
                        raw_completion(tuple(rows[j]),reasons[j],histories[j],seeds[i],version)
                        persisted.add(j)
                    elif (step+1)%128==0:
                        event('partial_batch_completion',stage_name=name,batch_slot=i,seed=seeds[i],
                              policy_version=version,prefix_sha256=fingerprint(histories[j]),
                              token_ids=tuple(rows[j]),sampled_log_probs=tuple(lps[j]))
                if all(reason is not None for reason in reasons):break
                attention=torch.cat((attention,active),-1)
                positions=(attention.sum(-1,keepdim=True)-1).clamp_min(0)
                if cache is None:raise ValueError('batch decoder did not return its native KV state')
                out=owner.model(input_ids=tokens,attention_mask=attention,position_ids=positions,
                    past_key_values=cache,use_cache=True,return_dict=True,**kwargs)
                cache=out.past_key_values;logits=out.logits[:,-1,:]
        except BaseException as exc:
            for j,i in enumerate(indices):
                if rows[j] and j not in persisted:
                    raw_completion(tuple(rows[j]),'interrupted:'+type(exc).__name__,histories[j],seeds[i],version)
                event('batch_stage_interrupted',stage_name=name,batch_slot=i,seed=seeds[i],
                      policy_version=version,prefix_sha256=fingerprint(histories[j]),
                      token_ids=tuple(rows[j]),sampled_log_probs=tuple(lps[j]),already_persisted=j in persisted)
            raise
        result={}
        for j,i in enumerate(indices):
            ids=tuple(rows[j]);after=state_hash(generators[i])
            event('batched_stage',batch_slot=i,stage_name=name,sampled_log_probs=lps[j],
                backend='hf_native_forward_batch_v1',seed=seeds[i],policy_version=version,
                prefix_sha256=fingerprint(histories[j]),native_token_ids=ids,finish_reason=reasons[j],
                rng_before_sha256=before[j],rng_after_sha256=after)
            if reasons[j]=='length' and limits[j]<budgets[j]:raise RolloutLengthError('batch context ceiling reached before action budget')
            result[i]=(ids,tuple(lps[j]),reasons[j],before[j],after)
        return result
    with torch.inference_mode(),owner.precision_context():
        first=stage(list(range(len(prefixes))),prefixes,[owner.max_new_tokens]*len(prefixes),'think',b'</think>')
        eligible=[];histories=[];budgets=[]
        for i in range(len(prefixes)):
            ids,lp,reason,_,_=first[i]
            match=re.fullmatch(r'(.*?)</think>\s*',fmt.decode(ids),re.DOTALL) if reason=='think_closed' else None
            if match and not any(tag in match.group(1) for tag in ('<think>','</think>','<summary>','</summary>','<tool_response>','</tool_response>')) and len(ids)<owner.max_new_tokens:
                eligible.append(i);histories.append(prefixes[i]+ids+fmt.control_ids);budgets.append(owner.max_new_tokens-len(ids))
        second=stage(eligible,histories,budgets,'summary',b'</summary>') if eligible else {}
        records=[]
        for i in range(len(prefixes)):
            a,ap,ar,r0,r1=first[i]
            b,bp,br,r2,r3=second.get(i,((),(),'not_started',None,None))
            if i in second and r1!=r2:raise RuntimeError('batch Summary reset its request RNG')
            g=fmt.record(a,b,ap,bp,ar,br,(r0,r1,r2,r3))
            event('batched_action_record',batch_slot=i,generation=g,seed=seeds[i],policy_version=version,
                prefix_sha256=fingerprint(prefixes[i]))
            records.append(g)
        return tuple(records)

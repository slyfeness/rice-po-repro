"""Independent environment-delimited Summary protocol; no sampled IDs rebuilt."""
import hashlib
import json
import re
from .contracts import Generation,InvalidGeneration,GenerationStage,Span,generation_action_mask
from .formats import NativeTagParser,AmbiguousBoundaryError,tagged_action
from .rollout import MalformedGenerationError,RolloutLengthError
from .audit import raw_completion,event,fingerprint

MODE='two_stage_env_summary_v1'


class UnrepresentableNativeToken(ValueError):
    """A sampled model-vocabulary row has no tokenizer text representation."""
    code='native_token_unmapped'


class NativeBytes:
    """Exact ByteLevel pieces, including literal added tokens. Cache per token ID."""
    def __init__(self,tokenizer):
        decoder=getattr(getattr(tokenizer,'backend_tokenizer',None),'decoder',None)
        if decoder is None or json.loads(decoder.__getstate__()).get('type')!='ByteLevel':
            raise ValueError('two-stage protocol requires verified native ByteLevel decoder')
        base=list(range(33,127))+list(range(161,173))+list(range(174,256))
        order=base+[b for b in range(256) if b not in base]
        chars=base+list(range(256,512-len(base)))
        self.inverse={chr(c):b for b,c in zip(order,chars,strict=True)}
        self.added=set(tokenizer.get_added_vocab().values())
        self.tokenizer=tokenizer;self.cache={}
    def piece(self,token):
        if token not in self.cache:
            p=self.tokenizer.convert_ids_to_tokens([token])[0]
            if p is None:
                raise UnrepresentableNativeToken('sampled token ID '+str(token)+' has no tokenizer representation')
            try:self.cache[token]=p.encode('utf8') if token in self.added else bytes(self.inverse[c] for c in p)
            except (KeyError,TypeError) as e:raise AmbiguousBoundaryError('unmapped native byte token') from e
        return self.cache[token]
    def join(self,ids):return b''.join(self.piece(t) for t in ids)


class TwoStageFormat:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.bytes=NativeBytes(tokenizer)
        # This is new deterministic ENVIRONMENT text, not a generated sequence.
        self.control_ids=tuple(tokenizer.encode('<summary>',add_special_tokens=False))
        if not self.control_ids or self.bytes.join(self.control_ids)!=b'<summary>' or self.decode(self.control_ids)!='<summary>':
            raise ValueError('environment Summary control does not decode exactly')
        probe=tuple(tokenizer.encode('</think>',add_special_tokens=False))
        if self.decode(probe+self.control_ids)!=self.decode(probe)+'<summary>':
            raise ValueError('environment control changes native concatenation')
        self.mapper=NativeTagParser(tokenizer,prefilled_think=True,malformed_policy='raise',
            structure_mode='tagged',boundary_rule='native_body_coverage_v1',allow_empty_reasoning=True)
    def decode(self,ids):
        return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    def record(self,think,summary,think_lp,summary_lp,think_reason,summary_reason,rngs=None):
        think,summary=tuple(think),tuple(summary)
        entered=bool(summary)
        control=self.control_ids if entered else ()
        ids=think+control+summary
        mask=(True,)*len(think)+(False,)*len(control)+(True,)*len(summary)
        lp=tuple(think_lp)+(None,)*len(control)+tuple(summary_lp)
        rngs=rngs or (None,None,None,None)
        stages=(GenerationStage('think',Span(0,len(think)),think_reason,rngs[0],rngs[1]),)
        if entered:stages+=(GenerationStage('summary',Span(len(think)+len(control),len(ids)),summary_reason,rngs[2],rngs[3]),)
        def invalid(category,reason):
            g=InvalidGeneration(ids,summary_reason if entered else think_reason,category,reason,mask,MODE,lp,stages)
            generation_action_mask(g)
            return g
        # ByteLevel permits sampling incomplete/invalid UTF-8. Replacement
        # decoding cannot supply an exact retrieval query or unique byte span.
        # Reject this real action; retain every native token and probability.
        try:self.bytes.join(ids).decode('utf8',errors='strict')
        except UnrepresentableNativeToken as exc:
            return invalid('native_token_unmapped',str(exc))
        except UnicodeDecodeError:
            return invalid('native_text_encoding','sampled native bytes are not valid UTF-8; no exact Summary available')
        if think_reason!='think_closed':
            return invalid('generation_length' if think_reason=='length' else 'tag_structure','Think did not sample its closing tag: '+think_reason)
        t=self.decode(think)
        tm=re.fullmatch(r'(.*?)</think>\s*',t,re.DOTALL)
        tags=('<think>','</think>','<summary>','</summary>','<tool_response>','</tool_response>')
        if tm is None or any(tag in tm.group(1) for tag in tags):
            return invalid('tag_structure','nested tags or non-whitespace beyond sampled Think close')
        if not entered:return invalid('generation_length','no remaining sampled-token budget for Summary')
        end=len(summary)
        if summary_reason=='eos':
            if summary[-1]!=self.tokenizer.eos_token_id:return invalid('tag_structure','Summary EOS identity mismatch')
            end-=1
        s=self.decode(summary[:end])
        sm=re.fullmatch(r'(.*?)</summary>\s*',s,re.DOTALL) if '</summary>' in s else None
        if summary_reason not in ('summary_closed','eos'):
            return invalid('generation_length','Summary exhausted sampled-token budget')
        if summary_reason=='summary_closed' and sm is None:
            return invalid('tag_structure','Summary close is not a complete terminal tag')
        body=sm.group(1) if sm else s
        if any(tag in body for tag in tags):return invalid('tag_structure','nested or duplicated Summary structure')
        if not body.strip():return invalid('empty_span','empty Summary body')
        full_ids=think+control+summary[:end];text=self.decode(full_ids)
        if text!=t+'<summary>'+s or self.bytes.join(full_ids)!=text.encode('utf8'):
            raise AmbiguousBoundaryError('two-stage complete native decode changed during concatenation')
        match=re.fullmatch(r'(.*?)</think>\s*<summary>(.*?)(?:</summary>\s*)?\Z',text,re.DOTALL)
        if match is None or match.group(2)!=body:raise AmbiguousBoundaryError('non-unique two-stage body')
        reasoning,span=self.mapper._body_byte_spans(full_ids,text,match)
        g=Generation(ids,reasoning,span,body,'stop',mask,MODE,lp,stages)
        generation_action_mask(g)
        return g


def generate_two_stage(owner,prefix,seed):
    """Two HF calls inside ONE seeded RNG scope, no probability transforms."""
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
    prefix=tuple(prefix);device=torch.device(owner.device)
    if owner.model.training or not prefix or type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('two-stage generation needs frozen eval model, native prefix and seed')
    replay=getattr(owner,'action_replay',None)
    cached=replay(prefix,seed,owner.policy_version) if replay else None
    if cached is not None:
        generation_action_mask(cached)
        if cached.protocol!=MODE:raise ValueError('saved action protocol differs')
        event('two_stage_saved_action_replay',generation=cached,recovery_replay=True,independent_sample=False)
        return cached
    if getattr(owner,'completion_replay',None):raise ValueError('legacy raw-completion replay cannot represent mixed two-stage records')
    fmt=owner.two_stage_format;version=owner.policy_version
    cuda=[device.index] if device.type=='cuda' else []
    def rng_hash():
        state=torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.random.get_rng_state()
        return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    def sample(history,budget,stage,marker):
        available=owner.max_context_tokens-len(history)
        if available<=0:raise RolloutLengthError('context exhausted before '+stage)
        limit=min(budget,available)
        class Observer(LogitsProcessor):
            pending=None
            values=[]
            def __init__(self):self.values=[];self.pending=None
            def __call__(self,input_ids,scores):
                if self.pending is not None:raise RuntimeError('sampling observer was not consumed')
                self.pending=torch.log_softmax(scores[0].float(),-1).detach()
                return scores  # Identity observer: exact same tensor, no mask/warp.
        observer=Observer()
        class Stop(StoppingCriteria):
            def __init__(self):self.previous=len(history);self.tail=b'';self.closed=False;self.unmapped=False;self.ids=[]
            def __call__(self,input_ids,scores,**kwargs):
                new=tuple(input_ids[0,self.previous:].tolist())
                if len(new)!=1 or observer.pending is None:raise RuntimeError('sampled-token observer alignment differs')
                observer.values.append(float(observer.pending[new[0]]));observer.pending=None
                self.ids.extend(new);self.previous+=1
                try:block=fmt.bytes.join(new)
                except UnrepresentableNativeToken:
                    self.unmapped=True
                    return torch.tensor([True],dtype=torch.bool,device=input_ids.device)
                self.closed=marker in self.tail+block
                self.tail=(self.tail+block)[-max(1,len(marker)-1):]
                return torch.tensor([self.closed],dtype=torch.bool,device=input_ids.device)
        stop=Stop();inputs=torch.tensor([history],dtype=torch.long,device=device)
        kwargs=dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),
            generation_config=owner._generation_recipe(limit),logits_processor=LogitsProcessorList([observer]),
            stopping_criteria=StoppingCriteriaList([stop]),synced_gpus=False)
        if owner._supports_use_model_defaults:kwargs['use_model_defaults']=False
        before=rng_hash()
        try:out=owner.model.generate(**kwargs)
        except BaseException as exc:
            if stop.ids:raw_completion(tuple(stop.ids),'interrupted:'+type(exc).__name__,history,seed,version)
            event('two_stage_stage_interrupted',stage_name=stage,native_token_ids=tuple(stop.ids),
                sampled_log_probs=observer.values,unconsumed_observer=observer.pending is not None,
                prefix_sha256=fingerprint(history),seed=seed,policy_version=version)
            raise
        after=rng_hash()
        if out.ndim!=2 or out.shape[0]!=1 or tuple(out[0,:len(history)].tolist())!=history:
            raise ValueError('two-stage model changed native prefix')
        ids=tuple(out[0,len(history):].tolist())
        if len(ids)!=len(observer.values) or not ids or len(ids)>limit:raise ValueError('native sampling trace is incomplete')
        eos=ids[-1]==owner.tokenizer.eos_token_id
        reason='native_token_unmapped' if stop.unmapped else ('think_closed' if stage=='think' else 'summary_closed') if stop.closed else 'eos' if eos else 'length'
        raw_completion(ids,reason,history,seed,version)
        event('two_stage_stage_saved',stage_name=stage,protocol=MODE,sampled_log_probs=observer.values,
            prefix_sha256=fingerprint(history),native_token_ids=ids,finish_reason=reason,
            rng_before_sha256=before,rng_after_sha256=after)
        if replay and hasattr(replay,'validate_stage'):
            replay.validate_stage(history,seed,version,stage,ids,tuple(observer.values),before,after)
        if owner.policy_version!=version:raise RuntimeError('policy changed between sampling stages')
        if len(ids)==available and available<budget and reason=='length':
            raise RolloutLengthError('native context ceiling exhausted during '+stage)
        return ids,tuple(observer.values),reason,before,after
    with torch.random.fork_rng(devices=cuda),torch.inference_mode(),owner.precision_context():
        torch.random.default_generator.manual_seed(seed)
        if device.type=='cuda':
            with torch.cuda.device(device):torch.cuda.manual_seed(seed)
        a,ap,ar,r0,r1=sample(prefix,owner.max_new_tokens,'think',b'</think>')
        b=();bp=();br='not_started';r2=r3=None
        # Refuse an invalid Think structure before inserting environment control.
        legal=re.fullmatch(r'(.*?)</think>\s*',fmt.decode(a),re.DOTALL) if ar=='think_closed' else None
        if legal and not any(t in legal.group(1) for t in ('<think>','</think>','<summary>','</summary>','<tool_response>','</tool_response>')):
            remaining=owner.max_new_tokens-len(a)
            if remaining>0:
                history=prefix+a+fmt.control_ids
                if fmt.decode(a+fmt.control_ids)!=fmt.decode(a)+'<summary>':raise AmbiguousBoundaryError('actual Summary control join differs')
                event('environment_summary_inserted',token_ids=fmt.control_ids,action_mask=(False,)*len(fmt.control_ids),protocol=MODE)
                b,bp,br,r2,r3=sample(history,remaining,'summary',b'</summary>')
                if r2!=r1:raise RuntimeError('RNG stream changed between Think and Summary')
        result=fmt.record(a,b,ap,bp,ar,br,(r0,r1,r2,r3))
        if replay and hasattr(replay,'validate_action'):
            replay.validate_action(prefix,seed,version,result)
        event('two_stage_action_record',generation=result)
        return result

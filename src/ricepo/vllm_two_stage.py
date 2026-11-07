"""Native two-stage vLLM adapter. No logits constraints, token repair or retries.

Stage seeds are a named independent stream, distinct from HF's shared generator.
The sampling law is unchanged; identical HF/vLLM sampled text is not required.
The injected engine returns every sampled ID, including EOS/stop-tag tokens.
"""
import math
import re
from .audit import stable_seed,raw_completion,event,fingerprint
from .two_stage import TwoStageFormat,UnrepresentableNativeToken,MODE
from .rollout import RolloutLengthError

BACKEND='vllm_ieee_two_stage_v1'
TAGS=('<think>','</think>','<summary>','</summary>','<tool_response>','</tool_response>')

# These are the only vLLM controls the RICE deployment benchmark is allowed to
# vary.  The sampling distribution, IEEE FP32 arithmetic, attention behavior
# and native-token protocol are deliberately *not* knobs.  Keeping this small
# schema makes an evaluated deployment recipe portable and prevents an
# apparently innocent throughput setting from silently changing the policy.
DEFAULT_RUNTIME = {
    'max_num_seqs': 64,
    'max_num_batched_tokens': 16384,
    'gpu_memory_utilization': .80,
    'enforce_eager': False,
    'enable_prefix_caching': True,
    'enable_chunked_prefill': True,
    'attention_backend': 'TRITON_ATTN',
}

# ``bf16_fast`` is deliberately a benchmark-only transport mode.  Its
# probability records can be checked, but it is not an approved replacement
# for the IEEE FP32 behavior policy used by the PPO path.
NUMERIC_MODES = {
    'ieee_fp32': {'dtype': 'float32', 'requires_triton_ieee': True, 'ppo_compatible': True},
    'bf16_fast': {'dtype': 'bfloat16', 'requires_triton_ieee': False, 'ppo_compatible': False},
    'tf32_pilot': {'dtype': 'float32', 'requires_triton_ieee': False, 'ppo_compatible': False},
}


def vllm_numeric_options(mode='ieee_fp32'):
    """Resolve an explicit numeric mode without silently changing PPO fidelity."""
    if mode not in NUMERIC_MODES:
        raise ValueError('unsupported numeric mode: '+repr(mode))
    return dict(NUMERIC_MODES[mode])


def vllm_runtime_options(options=None, *, numeric_mode='ieee_fp32'):
    """Validate benchmarkable scheduler settings without relaxing fidelity.

    ``None`` is the previously verified FP32 baseline.  Unknown keys are
    rejected rather than being accepted by vLLM with version-dependent
    defaults.  This function is intentionally import-light so static recipe
    expansion and CPU tests do not need vLLM installed.
    """
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError('vLLM runtime options must be an object')
    unknown = set(options).difference(DEFAULT_RUNTIME)
    if unknown:
        raise ValueError('unsupported vLLM runtime option(s): '+','.join(sorted(unknown)))
    resolved = {**DEFAULT_RUNTIME, **options}
    if type(resolved['max_num_seqs']) is not int or resolved['max_num_seqs'] <= 0:
        raise ValueError('max_num_seqs must be a positive integer')
    if type(resolved['max_num_batched_tokens']) is not int or resolved['max_num_batched_tokens'] <= 0:
        raise ValueError('max_num_batched_tokens must be a positive integer')
    memory = resolved['gpu_memory_utilization']
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not .5 <= float(memory) <= .95:
        raise ValueError('gpu_memory_utilization must be within [0.5, 0.95]')
    if (resolved['enforce_eager'] is not False or resolved['enable_prefix_caching'] is not True
            or resolved['enable_chunked_prefill'] is not True):
        raise ValueError('RICE benchmark requires CUDA Graphs, prefix caching and chunked prefill')
    numeric=vllm_numeric_options(numeric_mode)
    required_attention='TRITON_ATTN' if numeric['dtype']=='float32' else 'FLASH_ATTN'
    if resolved['attention_backend'] != required_attention:
        raise ValueError('numeric mode requires '+required_attention)
    resolved['gpu_memory_utilization'] = float(memory)
    return resolved

class VLLMTwoStage:
    def __init__(self,engine,tokenizer,policy_version_getter,max_new_tokens=4096,max_context_tokens=32768,backend=BACKEND):
        self.engine=engine;self.format=TwoStageFormat(tokenizer);self.version=policy_version_getter
        self.backend=backend
        self.max_new_tokens=max_new_tokens;self.max_context_tokens=max_context_tokens
        if type(max_new_tokens) is not int or max_new_tokens<=0:raise ValueError('positive action budget required')
    @property
    def policy_version(self):return self.version()
    def generate(self,prefix,*,seed):return self.generate_many([prefix],[seed])[0]
    def generate_many(self,prefixes,seeds):
        if not prefixes or len(prefixes)!=len(seeds):raise ValueError('aligned nonempty requests required')
        prefixes=[tuple(p) for p in prefixes];version=self.policy_version
        if any(not p or any(type(t) is not int or t<0 for t in p) for p in prefixes):raise ValueError('native prefix required')
        if any(type(s) is not int or not 0<=s<2**63 for s in seeds):raise ValueError('explicit independent seeds required')
        def stage(indices,histories,budgets,name):
            marker=('</think>' if name=='think' else '</summary>').encode()
            requests=[]
            for i,p,b in zip(indices,histories,budgets,strict=True):
                available=self.max_context_tokens-len(p)
                if available<=0:raise RolloutLengthError('vLLM context exhausted before '+name)
                requests.append(dict(prefix=p,budget=min(b,available),seed=stable_seed(seeds[i],self.backend,name),request_slot=i))
            outputs=self.engine.sample(requests,name,version)
            if len(outputs)!=len(requests):raise ValueError('engine omitted scheduled response')
            results=[]
            for r,o in zip(requests,outputs,strict=True):
                ids=tuple(o['token_ids']);lp=tuple(o['sampled_log_probs'])
                # Persist before text inspection, including unsupported vocabulary rows.
                raw_completion(ids,o['finish_reason'],r['prefix'],r['seed'],version)
                event('vllm_stage_raw',stage_name=name,policy_version=version,root_seed=seeds[r['request_slot']],
                      actual_stage_seed=r['seed'],prefix_sha256=fingerprint(r['prefix']),token_ids=ids,sampled_log_probs=lp,
                      engine_finish_reason=o['finish_reason'],sampling_backend=self.backend)
                if not ids or len(ids)>r['budget'] or len(lp)!=len(ids) or any(not math.isfinite(x) for x in lp):raise ValueError('incomplete native probability record')
                try:raw=self.format.bytes.join(ids)
                except UnrepresentableNativeToken:reason='native_token_unmapped'
                else:
                    reason=(name+'_closed') if marker in raw else 'eos' if ids[-1]==self.format.tokenizer.eos_token_id else 'length'
                if reason=='length' and len(ids)<r['budget']:raise ValueError('unexpected engine stop without native protocol endpoint')
                if reason=='length' and r['budget']<budgets[len(results)]:raise RolloutLengthError('vLLM native context ceiling exhausted')
                results.append((ids,lp,reason))
            if self.policy_version!=version:raise RuntimeError('policy changed during sampling')
            return results
        first=stage(list(range(len(prefixes))),prefixes,[self.max_new_tokens]*len(prefixes),'think')
        second={};active=[];histories=[];budgets=[]
        for i,(a,ap,ar) in enumerate(first):
            legal=re.fullmatch(r'(.*?)</think>\s*',self.format.decode(a),re.DOTALL) if ar=='think_closed' else None
            if legal and not any(t in legal.group(1) for t in TAGS) and len(a)<self.max_new_tokens:
                active.append(i);histories.append(prefixes[i]+a+self.format.control_ids);budgets.append(self.max_new_tokens-len(a))
                event('environment_summary_inserted',token_ids=self.format.control_ids,action_mask=(False,)*len(self.format.control_ids),protocol=MODE)
        if active:second=dict(zip(active,stage(active,histories,budgets,'summary'),strict=True))
        results=[]
        for i,(a,ap,ar) in enumerate(first):
            b,bp,br=second.get(i,((),(),'not_started'))
            result=self.format.record(a,b,ap,bp,ar,br)
            event('two_stage_action_record',generation=result,sampling_backend=self.backend)
            results.append(result)
        return tuple(results)

class VLLMEngine:
    """Single immutable local snapshot; restart this engine to switch weights.

    Stop strings terminate after sampling and keep the entire native stop token.
    Unrepresentable rows are stop IDs, never masked out of the distribution.
    """
    def __init__(self,model_path,tokenizer,policy_version,*,max_context_tokens=32768,runtime_options=None,
                 numeric_mode='ieee_fp32'):
        import os
        self.numeric=vllm_numeric_options(numeric_mode)
        if self.numeric['requires_triton_ieee'] and os.environ.get('TRITON_F32_DEFAULT')!='ieee':
            raise ValueError('IEEE Triton must be set before engine import')
        if numeric_mode=='tf32_pilot' and os.environ.get('TRITON_F32_DEFAULT')!='tf32':
            raise ValueError('TF32 pilot requires explicit pre-import Triton precision')
        from vllm import LLM
        self.version=policy_version
        self.runtime_options=vllm_runtime_options(runtime_options,numeric_mode=numeric_mode)
        self.llm=LLM(model=str(model_path),tokenizer=str(model_path),dtype=self.numeric['dtype'],tensor_parallel_size=1,
            max_model_len=max_context_tokens,max_num_seqs=self.runtime_options['max_num_seqs'],
            max_num_batched_tokens=self.runtime_options['max_num_batched_tokens'],
            gpu_memory_utilization=self.runtime_options['gpu_memory_utilization'],
            enforce_eager=self.runtime_options['enforce_eager'],
            enable_prefix_caching=self.runtime_options['enable_prefix_caching'],
            enable_chunked_prefill=self.runtime_options['enable_chunked_prefill'],hf_overrides={'sliding_window':None},
            generation_config='vllm',trust_remote_code=False,
            attention_config={'backend':self.runtime_options['attention_backend']})
        size=self.llm.llm_engine.model_config.hf_config.vocab_size
        self.unmapped=[i for i in range(size) if tokenizer.convert_ids_to_tokens(i) is None]
    def sample(self,requests,stage,policy):
        from vllm import SamplingParams
        if policy!=self.version:raise ValueError('engine weights are stale')
        params=[SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,max_tokens=r['budget'],seed=r['seed'],
            logprobs=0,detokenize=True,stop=['</think>' if stage=='think' else '</summary>'],
            stop_token_ids=self.unmapped,include_stop_str_in_output=True,skip_special_tokens=False,
            presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False) for r in requests]
        outputs=self.llm.generate([{'prompt_token_ids':list(r['prefix'])} for r in requests],params,use_tqdm=False)
        # Save every returned native completion BEFORE validating identities or
        # converting probability entries. An incomplete LP row must not erase
        # already generated tokens, including later rows in this same batch.
        event('vllm_engine_raw_response',stage_name=stage,policy_version=policy,
            requests=requests,outputs=[dict(prompt_token_ids=list(o.prompt_token_ids),completions=[
                dict(token_ids=list(c.token_ids),finish_reason=c.finish_reason) for c in o.outputs]) for o in outputs])
        rows=[]
        for r,o in zip(requests,outputs,strict=True):
            if tuple(o.prompt_token_ids)!=r['prefix'] or len(o.outputs)!=1:raise ValueError('engine changed request identity')
            c=o.outputs[0];ids=list(c.token_ids)
            rows.append(dict(token_ids=ids,sampled_log_probs=[float(d[t].logprob) for t,d in zip(ids,c.logprobs,strict=True)],finish_reason=c.finish_reason))
        return rows

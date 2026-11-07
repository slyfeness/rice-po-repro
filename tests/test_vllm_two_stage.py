"""No vLLM import or assets: scripted native transport contracts."""
import pytest
from test_two_stage_format import Tok
from ricepo.contracts import Generation,InvalidGeneration
from ricepo.vllm_two_stage import VLLMTwoStage


def test_runtime_options_only_expose_scheduler_controls_and_keep_ieee_protocol():
    from ricepo.vllm_two_stage import vllm_numeric_options,vllm_runtime_options
    resolved=vllm_runtime_options({'max_num_seqs':128,'max_num_batched_tokens':32768,
                                   'gpu_memory_utilization':.9})
    assert resolved['max_num_seqs']==128 and resolved['enable_prefix_caching'] is True
    with pytest.raises(ValueError,match='unsupported'):
        vllm_runtime_options({'dtype':'bfloat16'})
    with pytest.raises(ValueError,match='CUDA Graphs'):
        vllm_runtime_options({'enforce_eager':True})
    assert vllm_numeric_options('ieee_fp32')['dtype']=='float32'
    assert vllm_numeric_options('bf16_fast')['dtype']=='bfloat16'
    assert vllm_numeric_options('bf16_fast')['ppo_compatible'] is False
    fast=vllm_runtime_options({'attention_backend':'FLASH_ATTN'},numeric_mode='bf16_fast')
    assert fast['attention_backend']=='FLASH_ATTN'
    with pytest.raises(ValueError,match='requires FLASH_ATTN'):
        vllm_runtime_options({},numeric_mode='bf16_fast')
    with pytest.raises(ValueError,match='unsupported numeric mode'):
        vllm_numeric_options('tf32')

class Engine:
    def __init__(self,first,second):self.scripts=[first,second];self.calls=[]
    def sample(self,requests,stage,policy):
        self.calls.append((requests,stage,policy))
        script=self.scripts[len(self.calls)-1]
        return [dict(token_ids=script[:r['budget']],sampled_log_probs=[-1.]*min(len(script),r['budget']),
                     finish_reason='stop') for r in requests]

def build(a,b,budget=64):
    engine=Engine(a,b)
    return VLLMTwoStage(engine,Tok(),lambda:'p',budget,1000),engine

def test_native_control_mask_shared_budget_and_stage_seed():
    a=list(b'r</think>');b=list(b'words')+[1000]
    g,e=build(a,b,len(a)+len(b));result=g.generate((7,),seed=42)
    assert isinstance(result,Generation) and result.summary_text=='words'
    assert result.token_ids==tuple(a)+g.format.control_ids+tuple(b)
    assert sum(result.action_mask)==len(a)+len(b)
    assert e.calls[1][0][0]['budget']==len(b)
    assert e.calls[1][0][0]['prefix']==(7,)+tuple(a)+g.format.control_ids
    assert e.calls[0][0][0]['seed']!=e.calls[1][0][0]['seed']

def test_early_eos_never_invents_summary_or_resamples():
    g,e=build([1000],[]);r=g.generate((7,),seed=42)
    assert isinstance(r,InvalidGeneration) and r.token_ids==(1000,)
    assert len(e.calls)==1

def test_full_budget_consumed_in_think_has_no_environment_control():
    a=list(b'r</think>');g,e=build(a,[],len(a));r=g.generate((7,),seed=42)
    assert isinstance(r,InvalidGeneration) and r.token_ids==tuple(a)
    assert len(e.calls)==1 and all(r.action_mask)

def test_identical_independent_outputs_retained_and_seed_order_stable():
    a=list(b'r</think>');b=list(b's')+[1000]
    g,e=build(a,b);out=g.generate_many([(7,),(7,)],[42,43]);seeds=[x['seed'] for x in e.calls[0][0]]
    h,f=build(a,b);rev=h.generate_many([(7,),(7,)],[43,42])
    assert len(out)==2 and out[0].token_ids==out[1].token_ids
    assert seeds==[x['seed'] for x in f.calls[0][0]][::-1]

def test_extra_tokens_after_first_stop_rejected_without_slicing():
    g,e=build(list(b'r</think>x'),[])
    r=g.generate((7,),seed=42)
    assert isinstance(r,InvalidGeneration) and r.token_ids==tuple(b'r</think>x')
    assert len(e.calls)==1

def test_transport_missing_probability_or_response_rejected():
    g,e=build(list(b'r</think>'),[1000]);e.sample=lambda *args:[]
    with pytest.raises(ValueError,match='omitted'):g.generate((7,),seed=42)

def test_policy_change_between_stages_rejected():
    g,e=build(list(b'r</think>'),[1000]);v=['p'];g.version=lambda:v[0];old=e.sample
    def mutate(*args):
        result=old(*args);v[0]='changed';return result
    e.sample=mutate
    with pytest.raises(RuntimeError,match='policy changed'):g.generate((7,),seed=42)

def test_unmapped_token_is_retained_with_probability():
    from test_two_stage_format import MissingTokenTok
    from ricepo.two_stage import TwoStageFormat
    g,e=build(list(b'r</think>'),[ord('s'),1001]);g.format=TwoStageFormat(MissingTokenTok())
    r=g.generate((7,),seed=42)
    assert isinstance(r,InvalidGeneration) and r.category=='native_token_unmapped'
    assert r.token_ids[-1]==1001 and r.sampled_log_probs[-1]==-1. and r.action_mask[-1]

def test_engine_requires_ieee_before_any_vllm_import(monkeypatch):
    from ricepo.vllm_two_stage import VLLMEngine
    monkeypatch.delenv('TRITON_F32_DEFAULT',raising=False)
    with pytest.raises(ValueError,match='IEEE'):VLLMEngine('unused',Tok(),'p')

def test_raw_returned_ids_survive_missing_engine_logprobs(monkeypatch,tmp_path):
    import sys,json
    from types import SimpleNamespace
    from ricepo.audit import attempt
    from ricepo.vllm_two_stage import VLLMEngine
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kw:kw))
    engine=object.__new__(VLLMEngine);engine.version='p';engine.unmapped=[]
    completion=SimpleNamespace(token_ids=[1000],logprobs=None,finish_reason='stop')
    engine.llm=SimpleNamespace(generate=lambda *a,**k:[SimpleNamespace(prompt_token_ids=[7],outputs=[completion])])
    with pytest.raises(TypeError),attempt(tmp_path):
        engine.sample([dict(prefix=(7,),budget=4,seed=42)],'think','p')
    records=[json.loads(p.read_text()) for p in tmp_path.rglob('events/*.json')]
    raw=[r for r in records if r['kind']=='vllm_engine_raw_response']
    assert raw and raw[0]['outputs'][0]['completions'][0]['token_ids']==[1000]

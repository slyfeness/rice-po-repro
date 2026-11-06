"""Scripted CPU controller tests, not evidence of real-model format success."""
from contextlib import nullcontext
import pytest
from ricepo.hf_runtime import HFNativeGenerator
from ricepo.contracts import Generation,InvalidGeneration
from test_two_stage_format import Tok

def build(first,second,budget):
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    class Tokenizer(Tok):bos_token_id=None
    class Model:
        training=False
        def __init__(self):self.calls=[];self.draws=[];self.identity=[]
        def generate(self,*,input_ids,generation_config,logits_processor,stopping_criteria,**kwargs):
            script=first if not self.calls else second
            self.calls.append((tuple(input_ids[0].tolist()),generation_config.max_new_tokens))
            out=input_ids
            for token in script[:generation_config.max_new_tokens]:
                scores=torch.zeros((1,1101))
                returned=logits_processor(out,scores)
                self.identity.append(returned is scores)
                self.draws.append(float(torch.rand(())))
                out=torch.cat((out,torch.tensor([[token]])),1)
                done=stopping_criteria(out,None)
                if bool(done.any()) or token==1000:break
            return out
    model=Model()
    return HFNativeGenerator(model=model,tokenizer=Tokenizer(),device='cpu',policy_version_getter=lambda:'policy',
        max_new_tokens=budget,max_context_tokens=1000,prefilled_think=True,precision_context=nullcontext,
        summary_mode='two_stage_env_summary_v1',failure_semantics='invalid_action_terminal_zero_v1'),model

def test_two_calls_share_rng_and_total_model_budget_without_logit_changes():
    torch=pytest.importorskip('torch')
    a=tuple(b'r</think>');b=tuple(b'word')+(1000,)
    generator,model=build(a,b,len(a)+len(b))
    result=generator.generate((7,),seed=43)
    assert isinstance(result,Generation)
    assert model.calls[1][0]==(7,)+a+generator.two_stage_format.control_ids
    assert model.calls[1][1]==len(b)
    assert sum(result.action_mask)==len(a)+len(b)
    assert all(model.identity)
    assert model.draws==torch.rand(len(a)+len(b),generator=torch.Generator().manual_seed(43)).tolist()
    assert result.stages[0].rng_after_sha256==result.stages[1].rng_before_sha256

@pytest.mark.parametrize('first,second,budget,reason',[
    (tuple(b'raw')+(1000,),(),50,'Think'),
    (tuple(b'r</think>'),(1000,),50,'empty Summary'),
    (tuple(b'r</think>'),(),4,'Think')])
def test_eos_empty_summary_and_length_preserve_actual_failed_actions(first,second,budget,reason):
    generator,model=build(first,second,budget)
    result=generator.generate((7,),seed=44)
    assert isinstance(result,InvalidGeneration) and reason in result.reason
    assert sum(result.action_mask)<=budget
    if len(model.calls)==1:assert all(result.action_mask)

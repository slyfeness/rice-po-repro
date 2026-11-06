import unittest
from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path
import tempfile,json
import torch
from ricepo.batched_generation import generate_many
from ricepo.two_stage import TwoStageFormat,MODE
from ricepo.contracts import Generation,InvalidGeneration
from ricepo.audit import attempt
from test_two_stage_format import Tok


class Scripted(torch.nn.Module):
    def __init__(self):super().__init__();self.word=b's';self.calls=0
    def forward(self,input_ids,attention_mask,position_ids,past_key_values=None,**kw):
        self.calls+=1
        history=input_ids if past_key_values is None else torch.cat((past_key_values,input_ids),-1)
        logits=torch.full((len(history),1,1001),-1000.)
        for i,row in enumerate(history):
            ids=row[attention_mask[i].bool()].tolist();p=ids.index(200) if 200 in ids else ids.index(201)
            text=bytes(ids[p+1:])
            if b'<summary>' in text:
                tail=text.split(b'<summary>')[-1];script=self.word+b'</summary>'
            else:
                tail=text;script=b'r</think>' if ids[p]==200 else b'longer</think>'
            token=script[len(tail)] if len(tail)<len(script) else 1000
            logits[i,0,token]=1000.
        return SimpleNamespace(logits=logits,past_key_values=history)


def owner(limit=64):
    model=Scripted().eval();tok=Tok()
    return SimpleNamespace(model=model,tokenizer=tok,generation_protocol=MODE,
        two_stage_format=TwoStageFormat(tok),device='cpu',policy_version='actual-test-model',
        max_context_tokens=256,max_new_tokens=limit,precision_context=nullcontext,
        _generation_recipe=lambda n:SimpleNamespace(temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.))


class BatchTests(unittest.TestCase):
    def test_device_error_retains_all_returned_native_tokens(self):
        o=owner();original=o.model.forward
        def failing(*args,**kwargs):
            if o.model.calls>=3:raise RuntimeError('injected device error')
            return original(*args,**kwargs)
        o.model.forward=failing
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'injected device'):
                with attempt(Path(directory)):
                    generate_many(o,[(200,),(201,)],[41,42])
            events=[json.loads(p.read_text()) for p in Path(directory).glob('attempts/*/events/*.json')]
            raw=[e for e in events if e['kind']=='raw_completion']
            self.assertEqual(len(raw),2)
            self.assertTrue(all(len(e['token_ids'])==3 for e in raw))
            self.assertTrue(all(e['finish_reason'].startswith('interrupted:') for e in raw))

    def test_permutation_native_masks_rng_and_no_cache_after_model_change(self):
        o=owner();a=generate_many(o,[(200,),(201,)],[41,42]);b=generate_many(o,[(201,),(200,)],[42,41])
        self.assertEqual(a,tuple(reversed(b)))
        for g in a:
            self.assertIsInstance(g,Generation);self.assertEqual(g.summary_text,'s')
            self.assertEqual(g.stages[0].rng_after_sha256,g.stages[1].rng_before_sha256)
            self.assertEqual(g.action_mask.count(False),len(o.two_stage_format.control_ids))
        o.model.word=b'z';o.policy_version='changed-real-test-model'
        c=generate_many(o,[(200,)],[41])[0]
        self.assertEqual(c.summary_text,'z')

    def test_budget_failure_keeps_every_request_and_native_action(self):
        o=owner(limit=2);out=generate_many(o,[(200,),(201,)],[41,42])
        self.assertEqual(len(out),2)
        for g in out:
            self.assertIsInstance(g,InvalidGeneration)
            self.assertEqual(sum(g.action_mask),2)


if __name__=='__main__':unittest.main()

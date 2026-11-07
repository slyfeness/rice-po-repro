import unittest
from dataclasses import replace
from ricepo.contracts import Generation,InvalidGeneration,GenerationStage,Span,generation_action_mask
from ricepo.credit import trajectory_advantages,select_anchors
from ricepo.failure_zero import validate_trajectory
from test_failure_zero import runner,rollout

def mixed():
    return Generation((11,13,40,12,14),Span(0,1),Span(3,4),'actual summary','stop',
        action_mask=(True,True,False,True,True),protocol='two_stage_env_summary_v1',
        sampled_log_probs=(-1.,-1.,None,-2.,-1.),
        stages=(GenerationStage('think',Span(0,2),'think_closed'),GenerationStage('summary',Span(3,5),'eos')))

class MixedActionTests(unittest.TestCase):
    def test_legacy_is_all_actions_but_cannot_hide_environment_tokens(self):
        g=Generation((1,2),Span(0,1),Span(1,2),'s','stop')
        self.assertEqual(generation_action_mask(g),(True,True))
        with self.assertRaises(ValueError):generation_action_mask(replace(g,action_mask=(True,False)))

    def test_new_protocol_requires_explicit_mask_and_sampled_probabilities(self):
        g=mixed()
        self.assertEqual(generation_action_mask(g),g.action_mask)
        with self.assertRaises(ValueError):generation_action_mask(replace(g,action_mask=None))
        with self.assertRaises(ValueError):generation_action_mask(replace(g,sampled_log_probs=None))

    def test_rollout_uses_action_budget_and_credit_keeps_environment_zero(self):
        r=runner();r.rules=replace(r.rules,max_new_tokens=4)
        r.generator.generate=lambda *a,**k:mixed()
        t=rollout(r);validate_trajectory(t,5)
        self.assertEqual(t.generated_mask[1:6],mixed().action_mask)
        t2=replace(t,trajectory_id='other')
        b=trajectory_advantages([t,t2],epsilon=1e-8)
        self.assertTrue(all(v==0 for v,m in zip(b.advantages[0],b.generated_masks[0]) if not m))
        a=select_anchors([t],{'t':[1.]*len(t.token_ids)},top_k=1)[0]
        self.assertTrue(all(t.generated_mask[a.summary_span.start:a.summary_span.end]))

    def test_failure_after_environment_insertion_keeps_mixed_mask(self):
        r=runner();g=mixed()
        bad=InvalidGeneration(g.token_ids,'eos','empty_span','empty summary',g.action_mask,g.protocol,g.sampled_log_probs,g.stages)
        r.generator.generate=lambda *a,**k:bad
        t=rollout(r);validate_trajectory(t,5)
        self.assertEqual(t.generated_mask,(False,)+g.action_mask)
        self.assertEqual(t.final_reward,0)

if __name__=='__main__':unittest.main()

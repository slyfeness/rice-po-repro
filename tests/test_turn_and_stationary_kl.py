"""CPU regressions for the two-round diagnostic and its zero-signal control."""
import unittest

from ricepo.formats import PromptRenderer, action_closed
from ricepo.objective import exact_reference_kl


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True, return_dict=False)
        assert messages[0]["role"] == "user"
        self.text = "<｜User｜>" + messages[0]["content"] + "<｜Assistant｜><think>\n"
        return [1, 10, 11, 12]

    def decode(self, ids, **_):
        assert tuple(ids) == (10, 11, 12)
        return self.text


class Regression(unittest.TestCase):
    def test_observation_after_generated_eos_or_action_close(self):
        renderer = PromptRenderer(Tokenizer(), "prompt", observation_top_k=1,
                                  observation_token_limit=10, prefill_think=True)
        self.assertEqual(renderer.next_user_turn([], (88, 89)), (2, 10, 11, 12))
        self.assertEqual(renderer.next_user_turn([], (88, 2)), (10, 11, 12))

    def test_literal_close_inside_reasoning(self):
        self.assertFalse(action_closed("We should end with </summary>."))
        self.assertTrue(action_closed("reason</think>\n<summary>query</summary>"))

    def test_stationary_and_nonstationary_kl(self):
        import torch
        initial = torch.randn(4, 4096, generator=torch.Generator().manual_seed(42))
        logits = initial.clone().requires_grad_()
        (.001 * exact_reference_kl(logits, initial, chunk_size=2).sum()).backward()
        self.assertEqual(logits.grad.count_nonzero().item(), 0)
        reference = initial.clone()
        reference[:, 0] += 1
        logits.grad = None
        kl = exact_reference_kl(logits, reference, chunk_size=2)
        kl.sum().backward()
        p = logits.detach().double().softmax(-1)
        delta = logits.detach().double().log_softmax(-1) - reference.double().log_softmax(-1)
        expected = p * (delta - (p * delta).sum(-1, keepdim=True))
        torch.testing.assert_close(logits.grad.double(), expected, atol=1e-8, rtol=1e-4)
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_zero_signal_adamw_and_signed_policy_gradient(self):
        import torch
        from ricepo.objective import ppo_loss
        logits = torch.nn.Parameter(torch.tensor([[[0.2,-0.1]]]))
        frozen = logits.detach().clone()
        optimizer = torch.optim.AdamW([logits], lr=1e-6, weight_decay=0)
        current = logits.log_softmax(-1)[...,0]
        loss = ppo_loss(current,current.detach(),torch.zeros_like(current),
                        torch.ones_like(current,dtype=torch.bool),clip_epsilon=.2,kl_coefficient=.001,
                        reference_kl=exact_reference_kl(logits,frozen,chunk_size=1),reduction="sum")
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(logits,frozen))
        for advantage in (-1.0,1.0):
            action = torch.tensor([[0.]],requires_grad=True)
            value = ppo_loss(action,action.detach(),torch.full_like(action,advantage),
                             torch.ones_like(action,dtype=torch.bool),clip_epsilon=.2,kl_coefficient=0.,
                             reference_kl=torch.zeros_like(action),reduction="sum")
            value.backward()
            self.assertEqual(action.grad.item(),-advantage)

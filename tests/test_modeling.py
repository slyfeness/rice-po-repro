"""Native causal scoring fixtures; no checkpoints, downloads, or execution on import."""

import pytest

from ricepo.modeling import causal_token_scores, make_forward
from ricepo.training import UpdateItem


def fake_models(torch):
    class CausalTable(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([
                [0.0, 0.2, -0.1, 0.6],
                [0.3, -0.8, 0.2, 0.1],
                [-0.4, 0.7, 0.8, -0.3],
                [1.0, 0.3, -0.1, -0.5],
            ]) + offset * torch.tensor([0.0, 1.0, -1.0, 0.5]))
            self.calls = []

        def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
            from types import SimpleNamespace

            self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone(),
                               use_cache, return_dict, torch.is_grad_enabled()))
            return SimpleNamespace(logits=self.weight[input_ids])

    policy, reference = CausalTable(0.0), CausalTable(0.4)
    reference.eval().requires_grad_(False)
    return policy, reference


def score(policy, reference, *, chunk=2, ids=(1, 2, 3, 0, 2),
          mask=(False, True, True, False, True)):
    return causal_token_scores(
        policy, reference, ids, mask, device="cpu", logit_chunk_size=chunk,
        max_context_tokens=20, policy_call_kwargs={}, reference_call_kwargs={},
    )


def test_native_causal_alignment_full_vocabulary_kl_and_loss_masks():
    torch = pytest.importorskip("torch")
    policy, reference = fake_models(torch)
    log_probs, kl = score(policy, reference)
    assert tuple(log_probs.shape) == tuple(kl.shape) == (1, 5)
    for target_position, source_id, target_id in ((1, 1, 2), (2, 2, 3), (4, 0, 2)):
        current = torch.log_softmax(policy.weight[source_id], dim=-1)
        frozen = torch.log_softmax(reference.weight[source_id], dim=-1)
        expected_kl = (current.exp() * (current - frozen)).sum()
        assert log_probs[0, target_position].item() == pytest.approx(current[target_id].item())
        assert kl[0, target_position].item() == pytest.approx(expected_kl.item())
    assert log_probs[0, [0, 3]].tolist() == [0.0, 0.0]
    assert kl[0, [0, 3]].tolist() == [0.0, 0.0]
    assert policy.calls[0][0].tolist() == [[1, 2, 3, 0]]
    assert reference.calls[0][0].tolist() == [[1, 2, 3, 0]]
    assert policy.calls[0][2:] == (False, True, True)
    assert reference.calls[0][2:] == (False, True, False)


def test_chunks_preserve_policy_gradients_and_reference_is_detached():
    torch = pytest.importorskip("torch")
    small_policy, small_reference = fake_models(torch)
    whole_policy, whole_reference = fake_models(torch)
    small_scores, small_kl = score(small_policy, small_reference, chunk=1)
    whole_scores, whole_kl = score(whole_policy, whole_reference, chunk=10)
    (-small_scores.sum() + 0.2 * small_kl.sum()).backward()
    (-whole_scores.sum() + 0.2 * whole_kl.sum()).backward()
    assert torch.allclose(small_scores, whole_scores)
    assert torch.allclose(small_kl, whole_kl)
    assert torch.allclose(small_policy.weight.grad, whole_policy.weight.grad)
    assert small_policy.weight.grad.abs().sum().item() > 0
    assert small_reference.weight.grad is None
    # Row3 predicts masked tool token at native position3, so has no direct loss.
    assert small_policy.weight.grad[3].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_make_forward_calls_the_passed_wrapper_instead_of_unwrapping_module():
    torch = pytest.importorskip("torch")
    policy, reference = fake_models(torch)

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        @property
        def module(self):
            raise AssertionError("scoring must not bypass wrapper hooks via .module")

        def forward(self, **kwargs):
            self.calls += 1
            return self.inner(**kwargs)

    wrapped = Wrapper(policy)
    callback = make_forward(reference_model=reference, device="cpu", logit_chunk_size=2,
                            max_context_tokens=20, policy_call_kwargs={}, reference_call_kwargs={})
    item = UpdateItem("trajectory", "old-policy", (1, 2, 3), (False, True, True),
                      (0.0, 1.0, -1.0), (0.0, -1.5, -1.2))
    log_probs, kl = callback(wrapped, item)
    assert wrapped.calls == 1
    assert tuple(log_probs.shape) == tuple(kl.shape) == (1, 3)
    assert log_probs.requires_grad and kl.requires_grad


def test_exact_kl_alone_contributes_policy_gradients():
    torch = pytest.importorskip("torch")
    policy, reference = fake_models(torch)
    _, kl = score(policy, reference)
    kl.sum().backward()
    assert policy.weight.grad.abs().sum().item() > 0
    assert reference.weight.grad is None


def test_reference_must_be_separately_frozen_and_eval_before_forward():
    torch = pytest.importorskip("torch")
    policy, reference = fake_models(torch)
    reference.requires_grad_(True)
    with pytest.raises(ValueError, match="requires_grad=False"):
        score(policy, reference)
    assert policy.calls == reference.calls == []
    reference.requires_grad_(False).train()
    with pytest.raises(ValueError, match="eval mode"):
        score(policy, reference)
    with pytest.raises(ValueError, match="separate models"):
        score(policy, policy)


@pytest.mark.parametrize("ids,mask", [
    ((1, 2), (True, True)),
    ((1, 2), (False, False)),
    ((1, 2), (False,)),
    ((1, False), (False, True)),
    ((1, 2), (False, 1)),
])
def test_invalid_native_inputs_are_rejected_before_model_forward(ids, mask):
    torch = pytest.importorskip("torch")
    policy, reference = fake_models(torch)
    with pytest.raises(ValueError):
        score(policy, reference, ids=ids, mask=mask)
    assert policy.calls == reference.calls == []

"""Analytical fixtures for full-vocabulary entropy, KL, and PPO clipping.

PyTorch is optional; these tests skip when it is unavailable. They have not
been executed as part of the implementation-only handoff.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from ricepo.objective import (
    exact_reference_kl,
    ppo_loss,
    sampled_reference_kl,
    token_entropy_from_logits,
)


def test_entropy_uses_every_vocabulary_probability_and_preserves_token_axes():
    probabilities = torch.tensor([[[0.5, 0.25, 0.25], [1 / 3, 1 / 3, 1 / 3]]], dtype=torch.float64)
    entropy = token_entropy_from_logits(probabilities.log(), chunk_size=1)
    assert entropy.shape == (1, 2)
    assert entropy.tolist()[0] == pytest.approx([1.5 * math.log(2), math.log(3)])
    assert torch.allclose(entropy, token_entropy_from_logits(probabilities.log(), chunk_size=8))


def test_exact_kl_matches_categorical_formula_and_detaches_reference():
    logits = torch.tensor([[math.log(0.8), math.log(0.2)]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[math.log(0.5), math.log(0.5)]], dtype=torch.float64, requires_grad=True)
    penalty = exact_reference_kl(logits, reference, chunk_size=1)
    assert penalty.item() == pytest.approx(0.8 * math.log(1.6) + 0.2 * math.log(0.4))
    penalty.sum().backward()
    assert logits.grad is not None
    assert bool((logits.grad != 0).any())
    assert reference.grad is None
    assert exact_reference_kl(logits.detach(), logits.detach(), chunk_size=1).item() == pytest.approx(0.0)


def test_sampled_kl_estimators_are_explicit_and_not_per_token_exact_kl():
    current = torch.tensor([math.log(0.8)], dtype=torch.float64)
    reference = torch.tensor([math.log(0.5)], dtype=torch.float64)
    assert sampled_reference_kl(current, reference, estimator="log_ratio").item() == pytest.approx(math.log(1.6))
    estimate = sampled_reference_kl(current, reference, estimator="nonnegative").item()
    assert estimate == pytest.approx(0.625 - 1 + math.log(1.6))
    with pytest.raises(ValueError, match="estimator"):
        sampled_reference_kl(current, reference, estimator="exact")


def policy_fixture():
    current = torch.tensor([[0.3, 0.1, float("nan")], [0.12, 0.28, float("nan")]], dtype=torch.float64).log().requires_grad_()
    old = torch.full((2, 3), math.log(0.2), dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[2.0, 2.0, float("nan")], [-2.0, -2.0, float("nan")]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    kl = torch.tensor([[0.1, 0.2, float("nan")], [0.3, 0.4, float("nan")]], dtype=torch.float64)
    return current, old, advantages, mask, kl


@pytest.mark.parametrize("reduction, expected", [("sum", 1.5), ("token_mean", 0.375), ("sequence_mean", 0.75)])
def test_signed_ppo_clipping_masked_kl_and_explicit_reductions(reduction, expected):
    current, old, advantages, mask, kl = policy_fixture()
    loss = ppo_loss(
        current, old, advantages, mask,
        clip_epsilon=0.2, kl_coefficient=0.5, reference_kl=kl, reduction=reduction,
    )
    assert loss.item() == pytest.approx(expected)


def test_clipping_gradients_and_reference_inputs_do_not_train_old_policy_or_advantages():
    current, old, advantages, mask, kl = policy_fixture()
    loss = ppo_loss(
        current, old, advantages, mask,
        clip_epsilon=0.2, kl_coefficient=0.5, reference_kl=kl, reduction="sum",
    )
    loss.backward()
    assert current.grad[0].tolist() == pytest.approx([0, -1, 0])
    assert current.grad[1].tolist() == pytest.approx([0, 2.8, 0])
    assert old.grad is None
    assert advantages.grad is None


def test_nonzero_kl_requires_explicit_estimate_and_empty_masks_are_rejected():
    current, old, advantages, mask, _ = policy_fixture()
    kwargs = dict(clip_epsilon=0.2, kl_coefficient=0.5, reference_kl=None, reduction="sum")
    with pytest.raises(ValueError, match="explicit reference_kl"):
        ppo_loss(current, old, advantages, mask, **kwargs)
    with pytest.raises(ValueError, match="at least one generated"):
        ppo_loss(current, old, advantages, torch.zeros_like(mask), **kwargs)
    kwargs["kl_coefficient"] = 0.0
    assert ppo_loss(current, old, advantages, mask, **kwargs).item() == pytest.approx(1.0)


def test_bad_entropy_chunking_and_nonfinite_logits_are_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        token_entropy_from_logits(torch.zeros(1, 2), chunk_size=0)
    with pytest.raises(ValueError, match="finite"):
        token_entropy_from_logits(torch.tensor([[float("nan"), 1.0]]), chunk_size=1)

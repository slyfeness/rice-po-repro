"""CPU tensor fixtures authored for future execution; every model here is synthetic."""

import json
import math
from dataclasses import asdict

import pytest

from ricepo.training import UpdateItem, ppo_plan_update, ppo_update


def toy(torch, advantages):
    from types import SimpleNamespace
    actor = torch.nn.Linear(1, 1, bias=False)
    actor.weight.data.zero_()
    reference = torch.nn.Linear(1, 1, bias=False).eval().requires_grad_(False)
    reference.weight.data.zero_()

    def forward(model, item):
        z = model.weight.flatten()[0]
        current = torch.stack([-z, z]).log_softmax(dim=-1)
        with torch.no_grad():
            r = reference.weight.flatten()[0]
            frozen = torch.stack([-r, r]).log_softmax(dim=-1)
        kl = (current.exp() * (current - frozen)).sum()
        mask = torch.tensor([item.generated_mask])
        return current[1] * mask, kl * mask

    items = []
    for i, values in enumerate(advantages):
        n = len(values)
        items.append(UpdateItem(str(i), "old", (0,) + (1,) * n, (False,) + (True,) * n,
                                (0.,) + tuple(values), (0.,) + (-math.log(2),) * n))
    return SimpleNamespace(actor=actor, reference=reference, forward=forward, items=items,
                           optimizer=torch.optim.SGD(actor.parameters(), lr=.2))


def recipe(**changes):
    return {"ppo_epochs": 1, "optimizer_minibatch_size": 2, "microbatch_size": 1,
            "gradient_accumulation": 2, "loss_reduction": "token_mean", "clip_epsilon": .1,
            "max_grad_norm": 10., "actor_kl_coefficient": .1, **changes}


def update(case, settings):
    return ppo_plan_update(case.actor, case.optimizer, case.items, policy_version="old",
                           forward=case.forward, device="cpu", recipe=settings)


def test_one_old_equal_current_step_has_ratio_one_but_a_policy_gradient():
    torch = pytest.importorskip("torch")
    case = toy(torch, [(1., -1.), (2., 0.)])
    result = update(case, recipe())
    assert result["optimizer_steps"] == 1 and result["ratio_mean"] == pytest.approx(1.)
    assert result["outside_clip_fraction"] == result["effective_clip_fraction"] == 0
    assert result["grad_norm_before_clip"] > 0 and result["parameter_probe_changed"]
    assert (result["positive_advantages"], result["negative_advantages"], result["zero_advantages"]) == (2, 1, 1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reduction", ["sum", "token_mean", "sequence_mean"])
def test_accumulated_microgradients_match_original_whole_batch_reference(reduction):
    torch = pytest.importorskip("torch")
    values = [(1.,), (-.5, 1.5, 0.)]
    original, planned = toy(torch, values), toy(torch, values)
    baseline = ppo_update(original.actor, original.optimizer, original.items, policy_version="old",
                          forward=original.forward, device="cpu", clip_epsilon=.1, kl_coefficient=.1,
                          loss_reduction=reduction, max_grad_norm=10., distributed=False)
    result = update(planned, recipe(loss_reduction=reduction))
    torch.testing.assert_close(planned.actor.weight, original.actor.weight)
    assert result["normalized_loss"] == pytest.approx(baseline.normalized_loss)
    assert result["grad_norm_before_clip"] == pytest.approx(baseline.gradient_norm)


def test_multiple_steps_keep_old_probabilities_advantages_and_reference_fixed():
    torch = pytest.importorskip("torch")
    case = toy(torch, [(1.,), (1.,)])
    cached = [asdict(i) for i in case.items]
    frozen = case.reference.weight.detach().clone()
    result = update(case, recipe(ppo_epochs=2, optimizer_minibatch_size=1, gradient_accumulation=1))
    assert result["optimizer_steps"] == 4
    assert result["steps"][0]["ratio_mean"] == pytest.approx(1.)
    assert result["steps"][1]["ratio_mean"] > 1.1 and result["steps"][1]["effective_clip_fraction"] == 1
    assert result["outside_clip_fraction"] > 0
    torch.testing.assert_close(case.reference.weight, frozen)
    assert case.reference.weight.grad is None and [asdict(i) for i in case.items] == cached


def test_zero_advantages_identical_reference_and_no_momentum_can_leave_policy_unchanged():
    torch = pytest.importorskip("torch")
    case = toy(torch, [(0.,), (0., 0.)])
    result = update(case, recipe())
    assert result["optimizer_steps"] == 1 and result["zero_gradient_all_steps"]
    assert result["zero_advantages"] == 3 and not result["parameter_probe_changed"]


def test_outside_clip_with_wrong_advantage_sign_does_not_claim_effective_clipping():
    torch = pytest.importorskip("torch")
    case = toy(torch, [(1.,), (-1.,)])
    result = update(case, recipe(optimizer_minibatch_size=1, gradient_accumulation=1))
    assert result["steps"][1]["outside_clip_fraction"] == 1
    assert result["steps"][1]["effective_clip_fraction"] == 0


@pytest.mark.parametrize("changes", [{"microbatch_size": 2}, {"gradient_accumulation": 3},
                                     {"optimizer_minibatch_size": 3, "gradient_accumulation": 3}])
def test_unsupported_plan_fails_before_updating_actor(changes):
    torch = pytest.importorskip("torch")
    case = toy(torch, [(1.,), (1.,)])
    with pytest.raises(ValueError):
        update(case, recipe(**changes))
    assert case.actor.weight.item() == 0

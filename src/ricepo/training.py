"""PPO update primitives with explicit global normalization and native alignment.

This module does not create models or jobs. Models, optimizer and distributed
process group belong to the explicitly configured deployment factory.
"""

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from .objective import ppo_loss


@dataclass(frozen=True)
class UpdateItem:
    trajectory_id: str
    policy_version: str
    token_ids: tuple[int, ...]
    generated_mask: tuple[bool, ...]
    advantages: tuple[float, ...]
    old_log_probs: tuple[float, ...]


@dataclass(frozen=True)
class UpdateResult:
    optimizer_steps: int
    global_generated_tokens: int
    global_sequences: int
    normalized_loss: float
    gradient_norm: float


def _validate_items(items, policy_version):
    if not items or len({item.trajectory_id for item in items}) != len(items):
        raise ValueError("optimizer needs a nonempty batch of distinct trajectories")
    for item in items:
        n = len(item.token_ids)
        if not n or any(len(x) != n for x in (item.generated_mask, item.advantages, item.old_log_probs)):
            raise ValueError("update tensors must align with native token IDs")
        if item.policy_version != policy_version or item.generated_mask[0]:
            raise ValueError("wrong old policy or missing causal prefix")
        if any(type(value) is not bool for value in item.generated_mask) or not any(item.generated_mask):
            raise ValueError("expected boolean generated mask with at least one action")
        if any(not math.isfinite(x) for flag, a, old in zip(item.generated_mask, item.advantages, item.old_log_probs)
               if flag for x in (a, old)):
            raise ValueError("generated advantages and old log probabilities must be finite")


def ppo_update(model, optimizer, items: Sequence[UpdateItem], *, policy_version: str,
               forward: Callable, device, clip_epsilon: float, kl_coefficient: float,
               loss_reduction: str, max_grad_norm: float, distributed: bool) -> UpdateResult:
    """One whole-batch optimizer update.

    forward(model, item) -> (native-aligned current_log_probs, reference_kl).
    Both tensors have shape [1, len(item.token_ids)]. Reference KL must retain
    the gradient with respect to the current policy. Old log-probs, masks and
    advantages are detached. FSDP/DDP averages gradients across ranks; we scale
    each local token SUM by world_size/global_denominator to conserve the
    declared global objective. Equal per-rank sequence counts are required to
    keep synchronized forward/backward collectives aligned.
    """
    import torch
    import torch.distributed as dist
    if not items:
        raise ValueError("every rank needs a nonempty complete shard")
    if isinstance(max_grad_norm, bool) or not math.isfinite(max_grad_norm) or not 0 < max_grad_norm:
        raise ValueError("max_grad_norm must be finite and positive")
    if loss_reduction not in {"sum", "token_mean", "sequence_mean"}:
        raise ValueError("unsupported reduction")
    _validate_items(items, policy_version)
    if distributed and (not dist.is_initialized()):
        raise RuntimeError("distributed process group is not initialized")
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        from torch.distributed.fsdp import FullyShardedDataParallel
        if not isinstance(model, (DistributedDataParallel, FullyShardedDataParallel)):
            raise TypeError("distributed update requires a DDP/FSDP model with averaged gradients")
    world_size = dist.get_world_size() if distributed else 1
    if distributed:
        counts = [None] * world_size
        dist.all_gather_object(counts, len(items))
        if len(set(counts)) != 1:
            raise ValueError("equal shard lengths required for synchronized FSDP forwards")
        identities = [None] * world_size
        dist.all_gather_object(identities, [item.trajectory_id for item in items])
        flattened = [identity for shard in identities for identity in shard]
        if len(set(flattened)) != len(flattened):
            raise ValueError("duplicate trajectory across ranks")
    totals = torch.tensor([sum(sum(i.generated_mask) for i in items), len(items)],
                          dtype=torch.long, device=device)
    if distributed:
        dist.all_reduce(totals)
    denominator = {"sum": 1, "token_mean": int(totals[0]), "sequence_mean": int(totals[1])}[loss_reduction]
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    for item in items:
        current, kl = forward(model, item)
        old = torch.tensor([item.old_log_probs], dtype=torch.float32, device=device)
        advantage = torch.tensor([item.advantages], dtype=torch.float32, device=device)
        mask = torch.tensor([item.generated_mask], dtype=torch.bool, device=device)
        loss = ppo_loss(current, old, advantage, mask, clip_epsilon=clip_epsilon,
                        kl_coefficient=kl_coefficient, reference_kl=kl, reduction="sum")
        (loss * world_size / denominator).backward()
        loss_sum += loss.detach().double()
    if hasattr(model, "clip_grad_norm_"):
        norm = model.clip_grad_norm_(max_grad_norm)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite gradient; optimizer was not stepped")
    optimizer.step()
    if distributed:
        dist.all_reduce(loss_sum)
    return UpdateResult(1, int(totals[0]), int(totals[1]),
                        float(loss_sum / denominator), float(norm))


def _parameter_probe(model):
    """Small fixed coordinates from every trainable parameter, kept on its device."""
    import torch
    samples = []
    for p in model.parameters():
        if p.requires_grad and p.numel():
            value = p.detach()
            first = tuple(0 for _ in value.shape)
            middle = tuple((n - 1) // 2 for n in value.shape)
            last = tuple(n - 1 for n in value.shape)
            samples.append(torch.stack([value[first], value[middle], value[last]]).double())
    if not samples:
        raise ValueError("model has no trainable parameters")
    probe = torch.cat(samples)
    if not bool(torch.isfinite(probe).all()):
        raise FloatingPointError("nonfinite parameter probe")
    return probe


def ppo_plan_update(model, optimizer, items: Sequence[UpdateItem], *, policy_version: str,
                    forward: Callable, device, recipe, step_observer=None, probability_guard=None) -> dict:
    """Single-process PPO over frozen original trajectories in declared order.

    One optimizer step consumes one whole optimizer minibatch. Every sequence
    backward uses that minibatch's denominator; the weights and autocast
    context are refreshed by the supplied forward after every optimizer step.
    """
    import torch
    items = tuple(items)
    _validate_items(items, policy_version)
    epochs, micro = recipe["ppo_epochs"], recipe["microbatch_size"]
    mini, accum = (len(items) if recipe[k] == "full_batch" else recipe[k]
                   for k in ("optimizer_minibatch_size", "gradient_accumulation"))
    if type(micro) is not int or micro != 1:
        raise ValueError("single-process reference requires microbatch_size=1")
    if any(type(v) is not int or v < 1 for v in (epochs, mini, accum)) or mini != accum or len(items) % mini:
        raise ValueError("expected positive epochs and complete, evenly sized accumulated optimizer minibatches")
    reduction = recipe["loss_reduction"]
    if reduction not in {"sum", "token_mean", "sequence_mean"}:
        raise ValueError("unknown loss reduction")
    clip, coefficient, limit = (recipe[k] for k in ("clip_epsilon", "actor_kl_coefficient", "max_grad_norm"))
    if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if probability_guard is not None:
        if epochs != 1 or mini != len(items):
            raise ValueError("fused full-batch risk requires exactly one global optimizer step")
        probability_guard.begin(items, policy_version)
    first_probe = _parameter_probe(model)
    steps = []
    for epoch in range(epochs):
        for start in range(0, len(items), mini):
            group = items[start:start + mini]
            tokens = sum(sum(item.generated_mask) for item in group)
            denominator = {"sum": 1, "sequence_mean": len(group), "token_mean": tokens}[reduction]
            optimizer.zero_grad(set_to_none=True)
            # Token-weighted stats. Only detached scalar reductions survive each backward.
            loss_sum = kl_sum = ratio_sum = 0.
            outside = effective = positive = negative = 0
            ratio_min, ratio_max = math.inf, -math.inf
            for item in group:
                current, kl = forward(model, item)
                if probability_guard is not None:
                    try: probability_guard.observe(item, current)
                    except BaseException:
                        optimizer.zero_grad(set_to_none=True)
                        raise
                old = torch.tensor([item.old_log_probs], dtype=current.dtype, device=device)
                adv = torch.tensor([item.advantages], dtype=current.dtype, device=device)
                mask = torch.tensor([item.generated_mask], dtype=torch.bool, device=device)
                loss = ppo_loss(current, old, adv, mask, clip_epsilon=clip, kl_coefficient=coefficient,
                                reference_kl=kl, reduction="sum")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite optimizer loss")
                (loss / denominator).backward()
                with torch.no_grad():
                    ratio = (current[mask] - old[mask]).exp()
                    active = adv[mask]
                    above, below = ratio > 1 + clip, ratio < 1 - clip
                    outside += int((above | below).sum())
                    effective += int(((above & (active > 0)) | (below & (active < 0))).sum())
                    positive += int((active > 0).sum())
                    negative += int((active < 0).sum())
                    ratio_min = min(ratio_min, float(ratio.min()))
                    ratio_max = max(ratio_max, float(ratio.max()))
                    ratio_sum += float(ratio.double().sum())
                    kl_sum += float(kl[mask].double().sum())
                    loss_sum += float(loss.detach())
                del current, kl, loss
            if probability_guard is not None:
                try: probability_guard.finish()
                except BaseException:
                    optimizer.zero_grad(set_to_none=True)
                    raise
            if step_observer is not None:step_observer('before_clip')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
            from .audit import operation
            with operation("optimizer_minibatch_step",epoch=epoch,minibatch_start=start):
                optimizer.step()
            if step_observer is not None:step_observer('after_step')
            steps.append({"epoch": epoch, "minibatch_start": start, "generated_tokens": tokens,
                          "sequences": len(group), "loss_denominator": denominator,
                          "normalized_loss": loss_sum / denominator,
                          "policy_loss": (loss_sum - coefficient * kl_sum) / denominator,
                          "reference_kl": kl_sum / denominator, "reference_kl_token_mean": kl_sum / tokens,
                          "grad_norm_before_clip": float(norm), "ratio_min": ratio_min,
                          "ratio_max": ratio_max, "ratio_mean": ratio_sum / tokens,
                          "outside_clip_count": outside, "effective_clip_count": effective,
                          "outside_clip_fraction": outside / tokens, "effective_clip_fraction": effective / tokens,
                          "positive_advantages": positive, "negative_advantages": negative,
                          "zero_advantages": tokens - positive - negative})
    delta = (_parameter_probe(model) - first_probe).abs()
    total_tokens = sum(s["generated_tokens"] for s in steps)
    return {
        "optimizer_steps": len(steps), "steps": steps,
        "global_sequences": len(items), "global_generated_tokens": sum(sum(i.generated_mask) for i in items),
        **{k: math.fsum(s[k] for s in steps) / len(steps)
           for k in ("normalized_loss", "policy_loss", "reference_kl")},
        "grad_norm_before_clip": max(s["grad_norm_before_clip"] for s in steps),
        "ratio_min": min(s["ratio_min"] for s in steps), "ratio_max": max(s["ratio_max"] for s in steps),
        "ratio_mean": math.fsum(s["ratio_mean"] * s["generated_tokens"] for s in steps) / total_tokens,
        **{k: sum(s[k] for s in steps) for k in ("positive_advantages", "negative_advantages", "zero_advantages")},
        **{k + "_fraction": sum(s[k] for s in steps) / total_tokens
           for k in ("positive_advantages","negative_advantages","zero_advantages")},
        "outside_clip_fraction": sum(s["outside_clip_count"] for s in steps) / total_tokens,
        "effective_clip_fraction": sum(s["effective_clip_count"] for s in steps) / total_tokens,
        "parameter_probe_coordinates": "first, per-axis midpoint, last of every trainable tensor",
        "parameter_probe_changed": bool(delta.any()), "parameter_probe_max_abs_delta": float(delta.max()),
        "zero_gradient_all_steps": all(s["grad_norm_before_clip"] == 0 for s in steps),
    }


def wrap_fsdp(model, *, local_rank: int, world_size: int, mixed_precision):
    """Deployment helper; four-rank correctness is NOT claimed without testing."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError("declared and actual distributed world sizes differ")
    if world_size not in {2, 4} or torch.cuda.device_count() < world_size:
        raise RuntimeError("insufficient Slurm-visible devices for explicit topology")
    torch.cuda.set_device(local_rank)
    return FullyShardedDataParallel(model, device_id=local_rank,
                                    mixed_precision=mixed_precision, use_orig_params=True)

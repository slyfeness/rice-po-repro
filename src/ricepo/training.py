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
    if len({item.trajectory_id for item in items}) != len(items):
        raise ValueError("duplicate trajectory in optimizer shard")
    for item in items:
        n = len(item.token_ids)
        if not n or any(len(x) != n for x in (item.generated_mask, item.advantages, item.old_log_probs)):
            raise ValueError("update tensors must align with native token IDs")
        if item.policy_version != policy_version or item.generated_mask[0]:
            raise ValueError("wrong old policy or missing causal prefix")
        if any(type(value) is not bool for value in item.generated_mask):
            raise ValueError("generated masks must contain booleans")
        if not any(item.generated_mask):
            raise ValueError("empty generated sequence")
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

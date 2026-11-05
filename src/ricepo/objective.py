"""Optional PyTorch helpers for RICE-PO equations (1), (10), and (11).

Importing this module does not import torch or initialize a model/device.
Native token offsets must already align: a causal model's logit position t-1
predicts token t, so callers must shift logits before assigning token scores.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch import Tensor

LossReduction = Literal["sum", "token_mean", "sequence_mean"]
SampledKLEstimator = Literal["log_ratio", "nonnegative"]


def _torch() -> Any:
    """Load the optional tensor backend only when equations are evaluated."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("objective helpers require the optional PyTorch dependency") from exc
    return torch


def _floating_tensor(value: Tensor, name: str) -> None:
    """Check inputs to the token-distribution equations (1), (10), and (11)."""
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")


def _chunk_size(chunk_size: int) -> None:
    """Validate token-row chunking; vocabulary is never truncated in (1)/(11)."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a strictly positive integer")


def _logit_rows(logits: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Flatten only token axes, preserving the full vocabulary in (1)/(11)."""
    _floating_tensor(logits, "logits")
    if logits.ndim < 2 or logits.shape[-1] == 0 or logits.numel() == 0:
        raise ValueError("logits must have nonempty token and vocabulary axes")
    return logits.reshape(-1, logits.shape[-1]), tuple(logits.shape[:-1])


def _log_probabilities(logit_chunk: Tensor) -> Tensor:
    """Normalize full-vocabulary rows for (1)/(11), promoting one chunk at a time."""
    torch = _torch()
    if not bool(torch.isfinite(logit_chunk).all()):
        raise ValueError("logits must be finite")
    if logit_chunk.dtype in (torch.float16, torch.bfloat16):
        logit_chunk = logit_chunk.float()
    return torch.log_softmax(logit_chunk, dim=-1)


def token_entropy_from_logits(logits: Tensor, *, chunk_size: int) -> Tensor:
    """Full-vocabulary Shannon entropy in nats for each token history, (1).

    Input has shape [..., vocabulary]; output shape is [...]. Chunking is
    across token positions, never across a top-k subset of the vocabulary.
    This function retains gradients; rollout diagnostics can use no_grad.
    """
    torch = _torch()
    _chunk_size(chunk_size)
    rows, shape = _logit_rows(logits)
    values = []
    for start in range(0, rows.shape[0], chunk_size):
        log_probabilities = _log_probabilities(rows[start : start + chunk_size])
        values.append(-(log_probabilities.exp() * log_probabilities).sum(dim=-1))
    return torch.cat(values).reshape(shape)


def exact_reference_kl(
    logits: Tensor, reference_logits: Tensor, *, chunk_size: int
) -> Tensor:
    """Exact categorical KL(current || reference) over each full vocabulary, (11).

    Both tensors must describe the same histories and tokenizer vocabulary.
    Reference gradients are detached. This is conditional token KL; the
    rollout histories supply a Monte Carlo approximation to the trajectory
    expectation. It is not an exact enumeration of every possible trajectory.
    """
    torch = _torch()
    _chunk_size(chunk_size)
    rows, shape = _logit_rows(logits)
    reference_rows, reference_shape = _logit_rows(reference_logits)
    if shape != reference_shape or logits.shape != reference_logits.shape:
        raise ValueError("current and reference logits must have identical shapes")
    if rows.device != reference_rows.device:
        raise ValueError("current and reference logits must share a device")
    reference_rows = reference_rows.detach()
    values = []
    for start in range(0, rows.shape[0], chunk_size):
        current = _log_probabilities(rows[start : start + chunk_size])
        reference = _log_probabilities(reference_rows[start : start + chunk_size])
        values.append((current.exp() * (current - reference)).sum(dim=-1))
    return torch.cat(values).reshape(shape)


def sampled_reference_kl(
    log_probs: Tensor,
    reference_log_probs: Tensor,
    *,
    estimator: SampledKLEstimator,
) -> Tensor:
    """Explicit sampled approximations to the reference penalty in (11).

    log_ratio returns log pi(a)-log ref(a). nonnegative returns
    exp(log ref(a)-log pi(a)) - (log ref(a)-log pi(a)) - 1.
    Their expectations equal conditional KL only with actions sampled from
    the current policy and matching support. Old-policy rollout samples are
    generally biased without appropriate importance weighting. Neither option
    is the exact full-vocabulary KL implemented by exact_reference_kl.
    """
    torch = _torch()
    _floating_tensor(log_probs, "log_probs")
    _floating_tensor(reference_log_probs, "reference_log_probs")
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("sampled log-probability tensors must have identical shapes")
    if log_probs.device != reference_log_probs.device:
        raise ValueError("sampled log-probability tensors must share a device")
    if not bool(torch.isfinite(log_probs).all() and torch.isfinite(reference_log_probs).all()):
        raise ValueError("sampled log probabilities must be finite")
    delta = log_probs - reference_log_probs.detach()
    if estimator == "log_ratio":
        return delta
    if estimator == "nonnegative":
        return torch.expm1(-delta) + delta
    raise ValueError("estimator must be log_ratio or nonnegative")


def ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    generated_mask: Tensor,
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    reference_kl: Tensor | None,
    reduction: LossReduction,
) -> Tensor:
    """Negative clipped policy objective plus reference KL, (10)--(11).

    Inputs are [sequence, native_token] arrays. Old-policy probabilities and
    advantages are detached. reference_kl must be explicitly supplied whenever
    kl_coefficient is nonzero, using exact_reference_kl or a consciously chosen
    approximation. Tool/prompt/padding positions are excluded from both terms.

    sum adds every generated-token loss; token_mean divides by the number of
    generated tokens; sequence_mean averages each sequence's token *sum*,
    matching the empirical expectation over token sums in (11). Empty
    sequences are excluded from sequence_mean, and an entirely empty mask is
    rejected. The paper does not specify a numerical clipping coefficient;
    Appendix A.1 reports a KL coefficient of 0.001, supplied by the caller.
    """
    torch = _torch()
    for name, value in (
        ("log_probs", log_probs),
        ("old_log_probs", old_log_probs),
        ("advantages", advantages),
    ):
        _floating_tensor(value, name)
    if log_probs.ndim != 2 or old_log_probs.shape != log_probs.shape or advantages.shape != log_probs.shape:
        raise ValueError("policy probabilities and advantages must share [sequence, token] shape")
    if not isinstance(generated_mask, torch.Tensor) or generated_mask.dtype != torch.bool:
        raise ValueError("generated_mask must be a boolean tensor")
    if generated_mask.shape != log_probs.shape:
        raise ValueError("generated_mask must have the policy tensor shape")
    if any(value.device != log_probs.device for value in (old_log_probs, advantages, generated_mask)):
        raise ValueError("objective tensors must share a device")
    if not bool(generated_mask.any()):
        raise ValueError("objective requires at least one generated token")
    if isinstance(clip_epsilon, bool) or not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be finite and strictly between zero and one")
    if isinstance(kl_coefficient, bool) or not math.isfinite(kl_coefficient) or kl_coefficient < 0:
        raise ValueError("kl_coefficient must be finite and nonnegative")
    if reduction not in ("sum", "token_mean", "sequence_mean"):
        raise ValueError("unknown loss reduction")
    for value in (log_probs, old_log_probs, advantages):
        if not bool(torch.isfinite(value[generated_mask]).all()):
            raise ValueError("generated-token probabilities and advantages must be finite")
    current = log_probs.masked_fill(~generated_mask, 0)
    previous = old_log_probs.detach().masked_fill(~generated_mask, 0)
    advantage = advantages.detach().masked_fill(~generated_mask, 0)
    ratio = torch.exp(current - previous)
    if not bool(torch.isfinite(ratio[generated_mask]).all()):
        raise ValueError("importance ratio overflowed")
    surrogate = torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
    )
    token_loss = -surrogate
    if reference_kl is None:
        if kl_coefficient != 0:
            raise ValueError("a nonzero KL coefficient requires explicit reference_kl")
    else:
        _floating_tensor(reference_kl, "reference_kl")
        if reference_kl.shape != log_probs.shape or reference_kl.device != log_probs.device:
            raise ValueError("reference_kl must match policy shape and device")
        if not bool(torch.isfinite(reference_kl[generated_mask]).all()):
            raise ValueError("generated-token reference KL must be finite")
        token_loss = token_loss + kl_coefficient * reference_kl.masked_fill(~generated_mask, 0)
    token_loss = token_loss.masked_fill(~generated_mask, 0)
    if reduction == "token_mean":
        return token_loss.sum() / generated_mask.sum()
    if reduction == "sequence_mean":
        return token_loss.sum() / generated_mask.any(dim=1).sum()
    return token_loss.sum()

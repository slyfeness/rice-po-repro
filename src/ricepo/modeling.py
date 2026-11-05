"""Native-token causal scoring of caller-owned policy/reference models.

No tokenizer, model, optional framework, or distributed process is initialized
on import. A causal logit at position t-1 scores the native token at position t.
The two models must share the exact tokenizer vocabulary and token semantics;
equal vocabulary dimensions alone cannot establish that deployment invariant.

Each call forwards one complete sequence through each supplied model wrapper.
Logit-row chunking bounds temporary normalization work, but the model still
materializes full-sequence logits and retains the policy's training graph.
This does NOT guarantee long-context memory feasibility or implement streamed
training/FSDP sharding. There is no .module bypass or detached policy KV cache.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .objective import exact_reference_kl

if TYPE_CHECKING:
    from torch import Tensor

    from .training import UpdateItem


def _validate_kwargs(kwargs: Mapping[str, Any], label: str) -> dict[str, Any]:
    reserved = {
        "input_ids", "inputs_embeds", "attention_mask", "position_ids",
        "cache_position", "past_key_values", "use_cache", "return_dict", "labels",
    }
    if reserved.intersection(kwargs):
        raise ValueError(f"{label} cannot override native-token causal alignment arguments")
    return dict(kwargs)


def causal_token_scores(
    policy_model: Any,
    reference_model: Any,
    token_ids: Sequence[int],
    generated_mask: Sequence[bool],
    *,
    device: Any,
    logit_chunk_size: int,
    max_context_tokens: int,
    policy_call_kwargs: Mapping[str, Any],
    reference_call_kwargs: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """Return [1,N] sampled policy log-probabilities and exact KL(policy||ref).

    Index zero has no causal predecessor and is explicitly zero padded. Prompt
    and tool positions are also zero and disconnected from direct loss terms;
    their native IDs remain in context for every subsequent actor prediction.
    Gradients through generated-token policy scores and KL are preserved when
    the caller enables autograd. Reference parameters must already be frozen
    and its model in eval mode; no_grad and detach provide additional isolation.

    All limits and model call options are explicit. No context is truncated and
    no token sequence is decoded, re-encoded, sampled, or reformatted here.
    """
    if type(logit_chunk_size) is not int or logit_chunk_size <= 0:
        raise ValueError("logit_chunk_size must be an explicit positive integer")
    if type(max_context_tokens) is not int or max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be an explicit positive integer")
    ids, mask_values = tuple(token_ids), tuple(generated_mask)
    if len(ids) < 2 or len(ids) != len(mask_values):
        raise ValueError("native IDs and mask must align and include a causal prefix")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("token_ids must be nonnegative native integer IDs")
    if any(type(flag) is not bool for flag in mask_values):
        raise ValueError("generated_mask must contain booleans")
    if mask_values[0] or not any(mask_values):
        raise ValueError("a non-generated first token and at least one generated token are required")
    if len(ids) > max_context_tokens:
        raise ValueError("trajectory exceeds explicit context limit; truncation is forbidden")
    if policy_model is reference_model:
        raise ValueError("policy and frozen reference must be separate models")
    if getattr(reference_model, "training", True):
        raise ValueError("reference model must already be in eval mode")
    if any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise ValueError("reference model parameters must already have requires_grad=False")
    policy_kwargs = _validate_kwargs(policy_call_kwargs, "policy_call_kwargs")
    reference_kwargs = _validate_kwargs(reference_call_kwargs, "reference_call_kwargs")

    # Optional dependency is loaded only during an explicitly requested forward.
    import torch

    inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    # Call the supplied wrapper itself, preserving DDP/FSDP hooks and gradients.
    policy_output = policy_model(
        input_ids=inputs, attention_mask=attention, use_cache=False,
        return_dict=True, **policy_kwargs,
    )
    logits = policy_output.logits
    with torch.no_grad():
        reference_output = reference_model(
            input_ids=inputs, attention_mask=attention, use_cache=False,
            return_dict=True, **reference_kwargs,
        )
        reference_logits = reference_output.logits.detach()
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(ids) - 1):
        raise ValueError("policy logits must cover every native causal predecessor")
    if not logits.is_floating_point() or not reference_logits.is_floating_point():
        raise ValueError("policy and reference logits must be floating-point tensors")
    if logits.shape[-1] == 0 or reference_logits.shape != logits.shape:
        raise ValueError("policy and reference must return identical full-vocabulary shapes")
    if logits.device != reference_logits.device or logits.device != inputs.device:
        raise ValueError("policy/reference logits and native inputs must share the explicit device")
    targets = torch.tensor(ids[1:], dtype=torch.long, device=logits.device)
    if bool((targets >= logits.shape[-1]).any()):
        raise ValueError("native target token ID exceeds the shared model vocabulary")
    mask = torch.tensor(mask_values[1:], dtype=torch.bool, device=logits.device)
    score_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    reference_dtype = (torch.float32 if reference_logits.dtype in (torch.float16, torch.bfloat16)
                       else reference_logits.dtype)
    kl_dtype = torch.promote_types(score_dtype, reference_dtype)
    sampled_chunks = [torch.zeros(1, dtype=score_dtype, device=logits.device)]
    kl_chunks = [torch.zeros(1, dtype=kl_dtype, device=logits.device)]
    for start in range(0, len(ids) - 1, logit_chunk_size):
        stop = min(start + logit_chunk_size, len(ids) - 1)
        positions = torch.nonzero(mask[start:stop], as_tuple=False).flatten()
        sampled = torch.zeros(stop - start, dtype=score_dtype, device=logits.device)
        divergence = torch.zeros(stop - start, dtype=kl_dtype, device=logits.device)
        if positions.numel():
            policy_rows = logits[0, start:stop].index_select(0, positions)
            reference_rows = reference_logits[0, start:stop].index_select(0, positions)
            if not bool(torch.isfinite(policy_rows).all()):
                raise ValueError("generated-token policy logits must be finite")
            log_probabilities = torch.log_softmax(policy_rows.to(score_dtype), dim=-1)
            selected_targets = targets[start:stop].index_select(0, positions)
            selected_scores = log_probabilities.gather(1, selected_targets[:, None]).squeeze(1)
            selected_kl = exact_reference_kl(
                policy_rows, reference_rows, chunk_size=logit_chunk_size,
            )
            # index_copy remains differentiable with respect to the policy rows.
            sampled = sampled.index_copy(0, positions, selected_scores)
            divergence = divergence.index_copy(0, positions, selected_kl)
        sampled_chunks.append(sampled)
        kl_chunks.append(divergence)
    return torch.cat(sampled_chunks)[None, :], torch.cat(kl_chunks)[None, :]


def make_forward(
    *,
    reference_model: Any,
    device: Any,
    logit_chunk_size: int,
    max_context_tokens: int,
    policy_call_kwargs: Mapping[str, Any],
    reference_call_kwargs: Mapping[str, Any],
) -> Callable[[Any, UpdateItem], tuple[Tensor, Tensor]]:
    """Build training.ppo_update's forward(model, UpdateItem) callback.

    The policy wrapper comes from ppo_update on each call; the factory never
    captures or unwraps a potentially different policy module. No forward,
    optional-framework import, model mutation, or allocation occurs here.
    """
    policy_kwargs = _validate_kwargs(policy_call_kwargs, "policy_call_kwargs")
    reference_kwargs = _validate_kwargs(reference_call_kwargs, "reference_call_kwargs")

    def forward(model: Any, item: UpdateItem) -> tuple[Tensor, Tensor]:
        return causal_token_scores(
            model, reference_model, item.token_ids, item.generated_mask,
            device=device, logit_chunk_size=logit_chunk_size,
            max_context_tokens=max_context_tokens,
            policy_call_kwargs=policy_kwargs, reference_call_kwargs=reference_kwargs,
        )

    return forward

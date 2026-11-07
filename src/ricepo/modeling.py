"""Native-token causal scoring of caller-owned policy/reference models.

No tokenizer, model, optional framework, or distributed process is initialized
on import. A causal logit at position t-1 scores the native token at position t.
The two models must share the exact tokenizer vocabulary and token semantics;
equal vocabulary dimensions alone cannot establish that deployment invariant.

The default forwards the sequence through each supplied model wrapper and
materializes full-sequence logits. Opt-in action_positions_v1 supports bare
Qwen2 only: it retains the same Transformer context and training graph but
projects selected causal predecessor states through the original full-vocabulary
head. Neither path guarantees long-context memory feasibility or implements
FSDP sharding. There is no .module bypass or detached policy KV cache.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from contextlib import nullcontext

from .objective import _conditional_kl, _log_probabilities, exact_reference_kl


def enable_eval_layer_checkpointing(model):
    """Recompute decoder layers while preserving eval/dropout and context gradients."""
    import torch
    from torch.utils.checkpoint import checkpoint
    if model.training:raise ValueError('eval layer checkpointing requires eval mode')
    layers=getattr(getattr(model,'model',None),'layers',None)
    if layers is None:raise ValueError('model does not expose supported decoder layers')
    if hasattr(model,'_rice_checkpoint_statistics'):raise ValueError('checkpoint wrappers already installed')
    statistics=dict(mode='eval_layers_v1',checkpoint_forwards=0,layer_executions=0)
    for layer in layers:
        original=layer.forward
        def wrap(fn):
            def forward(*args,**kwargs):
                if not torch.is_grad_enabled():return fn(*args,**kwargs)
                if kwargs.get('past_key_values') is not None or kwargs.get('use_cache',False):
                    raise ValueError('training recomputation cannot reuse an inference KV cache')
                statistics['checkpoint_forwards']+=1
                def recompute(*a,**k):
                    statistics['layer_executions']+=1
                    return fn(*a,**k)
                return checkpoint(recompute,*args,use_reentrant=False,**kwargs)
            return forward
        layer.forward=wrap(original)
    model._rice_checkpoint_statistics=statistics
    return statistics

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


def _validate_projection(mode):
    if mode not in ('full_sequence', 'action_positions_v1'):
        raise ValueError('unsupported LM-head projection mode')


def _action_logits(model, inputs, attention, positions, kwargs):
    # Opt-in bare-model capability, never an implicit DDP/FSDP/PEFT bypass.
    if (type(model).__name__ != 'Qwen2ForCausalLM'
            or getattr(getattr(model, 'config', None), 'model_type', None) != 'qwen2'
            or kwargs):
        raise ValueError('action_positions_v1 requires bare Qwen2ForCausalLM and empty call kwargs')
    output = model.model(input_ids=inputs, attention_mask=attention, use_cache=False,
                         return_dict=True, output_hidden_states=False, output_attentions=False)
    hidden = output.last_hidden_state.index_select(1, positions)
    return model.lm_head(hidden)


def _native_scores(
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
    compact: bool,
    projection_mode: str = 'full_sequence',
    measure: Any = None,
) -> tuple[Tensor, Tensor | None]:
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
    _validate_projection(projection_mode)
    timed = measure if measure is not None else lambda name: nullcontext()
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
    if reference_model is not None and getattr(reference_model, "training", True):
        raise ValueError("reference model must already be in eval mode")
    if reference_model is not None and any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise ValueError("reference model parameters must already have requires_grad=False")
    policy_kwargs = _validate_kwargs(policy_call_kwargs, "policy_call_kwargs")
    reference_kwargs = _validate_kwargs(reference_call_kwargs, "reference_call_kwargs")

    # Optional dependency is loaded only during an explicitly requested forward.
    import torch

    # A causal actor cannot use observations beyond its final scored action.
    # Keep every earlier native token, including masked intermediate observations.
    horizon = max(i for i, scored in enumerate(mask_values) if scored) if compact else len(ids) - 1
    inputs = torch.tensor([ids[:horizon]], dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    action_positions = torch.tensor([i-1 for i in range(1,horizon+1) if mask_values[i]],
                                    dtype=torch.long, device=device)
    selected = projection_mode == 'action_positions_v1'
    def project(model, kwargs):
        if selected:
            return _action_logits(model, inputs, attention, action_positions, kwargs)
        # Default calls the supplied wrapper, preserving distributed hooks.
        return model(input_ids=inputs, attention_mask=attention, use_cache=False,
                     return_dict=True, **kwargs).logits
    with timed('actor_forward'):
        logits = project(policy_model, policy_kwargs)
    reference_logits = None
    if reference_model is not None:
        with torch.no_grad(), timed('reference_forward'):
            reference_logits = project(reference_model, reference_kwargs).detach()
        if (reference_logits.shape != logits.shape or not reference_logits.is_floating_point()
                or reference_logits.device != logits.device):
            raise ValueError("policy and reference must return identical full-vocabulary shapes on one device")
    expected_rows = len(action_positions) if selected else horizon
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, expected_rows):
        raise ValueError("policy logits must cover every native causal predecessor")
    if not logits.is_floating_point() or logits.shape[-1] == 0 or logits.device != inputs.device:
        raise ValueError("expected full-vocabulary floating-point logits on the explicit device")
    targets = torch.tensor(ids[1:], dtype=torch.long, device=logits.device)
    if bool((targets >= logits.shape[-1]).any()):
        raise ValueError("native target token ID exceeds the shared model vocabulary")
    mask = torch.tensor(mask_values[1:], dtype=torch.bool, device=logits.device)
    score_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    reference_dtype = score_dtype if reference_logits is None else reference_logits.dtype
    if reference_dtype in (torch.bfloat16, torch.float16):
        reference_dtype = torch.float32
    kl_dtype = torch.promote_types(score_dtype, reference_dtype)
    sampled_chunks = [torch.zeros(1, dtype=score_dtype, device=logits.device)]
    kl_chunks = [torch.zeros(1, dtype=kl_dtype, device=logits.device)] if reference_model is not None else []
    row_offset = 0
    for start in range(0, horizon, logit_chunk_size):
        stop = min(start + logit_chunk_size, horizon)
        positions = torch.nonzero(mask[start:stop], as_tuple=False).flatten()
        sampled = torch.zeros(stop - start, dtype=score_dtype, device=logits.device)
        if reference_logits is not None:
            divergence = torch.zeros(stop - start, dtype=kl_dtype, device=logits.device)
        if positions.numel():
            count = positions.numel()
            policy_rows = (logits[0, row_offset:row_offset+count] if selected
                           else logits[0, start:stop].index_select(0, positions))
            with timed('policy_normalization'):
                log_probabilities = _log_probabilities(policy_rows)
            selected_targets = targets[start:stop].index_select(0, positions)
            selected_scores = log_probabilities.gather(1, selected_targets[:, None]).squeeze(1)
            # index_copy remains differentiable with respect to the policy rows.
            sampled = sampled.index_copy(0, positions, selected_scores)
            if reference_logits is not None:
                reference_rows = (reference_logits[0, row_offset:row_offset+count] if selected
                                  else reference_logits[0, start:stop].index_select(0, positions))
                with timed('exact_KL'):
                    if compact:
                        ref_lp = _log_probabilities(reference_rows)
                        selected_kl = _conditional_kl(log_probabilities, ref_lp)
                    else:
                        selected_kl = exact_reference_kl(policy_rows, reference_rows, chunk_size=logit_chunk_size)
                divergence = divergence.index_copy(0, positions, selected_kl)
            row_offset += count
        sampled_chunks.append(sampled)
        if reference_logits is not None:
            kl_chunks.append(divergence)
    tail = len(ids) - horizon - 1
    sampled = torch.nn.functional.pad(torch.cat(sampled_chunks), (0, tail))[None, :]
    kl = torch.nn.functional.pad(torch.cat(kl_chunks), (0, tail))[None, :] if kl_chunks else None
    return sampled, kl


def causal_token_scores(policy_model, reference_model, token_ids, generated_mask, *, device, logit_chunk_size,
                       max_context_tokens, policy_call_kwargs, reference_call_kwargs,
                       projection_mode='full_sequence') -> tuple[Tensor, Tensor]:
    """Full-sequence comparator with independent full-vocabulary KL normalization."""
    if reference_model is None:
        raise ValueError("causal_token_scores requires a frozen reference")
    return _native_scores(policy_model, reference_model, token_ids, generated_mask, device=device,
                          logit_chunk_size=logit_chunk_size, max_context_tokens=max_context_tokens,
                          policy_call_kwargs=policy_call_kwargs, reference_call_kwargs=reference_call_kwargs,
                          compact=False, projection_mode=projection_mode)


def sampled_old_log_probs(policy, token_ids, generated_mask, *, device, logit_chunk_size, max_context_tokens,
                          policy_call_kwargs, projection_mode='full_sequence', measure=None) -> Tensor:
    """Cache detached sampled action log probabilities with one policy-only forward."""
    import torch
    with torch.no_grad():
        scores, _ = _native_scores(policy, None, token_ids, generated_mask, device=device,
                                   logit_chunk_size=logit_chunk_size, max_context_tokens=max_context_tokens,
                                   policy_call_kwargs=policy_call_kwargs, reference_call_kwargs={}, compact=True,
                                   projection_mode=projection_mode, measure=measure)
    return scores


def make_forward(
    *,
    reference_model: Any,
    device: Any,
    logit_chunk_size: int,
    max_context_tokens: int,
    policy_call_kwargs: Mapping[str, Any],
    reference_call_kwargs: Mapping[str, Any],
    projection_mode: str = 'full_sequence',
    measure: Any = None,
    saved_tensor_policy: str = 'device',
) -> Callable[[Any, UpdateItem], tuple[Tensor, Tensor]]:
    """Build training.ppo_update's forward(model, UpdateItem) callback.

    The policy wrapper comes from ppo_update on each call; the factory never
    captures or unwraps a potentially different policy module. No forward,
    optional-framework import, model mutation, or allocation occurs here.
    """
    _validate_projection(projection_mode)
    if saved_tensor_policy not in ('device','cpu_exact_v1'):
        raise ValueError('unknown saved tensor residency policy')
    policy_kwargs = _validate_kwargs(policy_call_kwargs, "policy_call_kwargs")
    reference_kwargs = _validate_kwargs(reference_call_kwargs, "reference_call_kwargs")
    if reference_model is None:
        raise ValueError("PPO scoring requires a separate frozen reference")

    def forward(model: Any, item: UpdateItem) -> tuple[Tensor, Tensor]:
        # Official autograd saved-tensor hooks preserve the graph and tensor
        # values; only residency changes. This is neither a detached context
        # nor a sampled/partial-vocabulary objective. Unpacking happens during
        # backward after this scope exits. Existing decoder recomputation stays.
        scope=nullcontext()
        if saved_tensor_policy=='cpu_exact_v1':
            import torch
            scope=torch.autograd.graph.save_on_cpu(pin_memory=False)
        with scope:
            return _native_scores(
                model, reference_model, item.token_ids, item.generated_mask,
                device=device, logit_chunk_size=logit_chunk_size,
                max_context_tokens=max_context_tokens,
                policy_call_kwargs=policy_kwargs, reference_call_kwargs=reference_kwargs, compact=True,
                projection_mode=projection_mode, measure=measure,
            )

    return forward

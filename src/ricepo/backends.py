"""Optional adapters. Constructing a model, tokenizer, or vLLM engine is caller-owned.

There are no optional-framework imports, downloads, or engine initializations at
module import. Prompt IDs and parser output remain native token sequences.
"""

from collections.abc import Callable, Mapping, Sequence
from math import nan
from typing import Any

from .contracts import Generation, Trajectory
from .rollout import MalformedGenerationError, PolicyVersionError, RolloutLengthError, _tokens


class BackendContractError(RuntimeError):
    """An explicitly configured backend returned an incompatible result."""


class VLLMGenerator:
    def __init__(
        self,
        *,
        engine: Any,
        policy_version_getter: Callable[[], str],
        sampling_params_factory: Callable[..., Any],
        sampling_settings: Mapping[str, Any],
        generation_parser: Callable[[tuple[int, ...], str], Generation],
        engine_generate_kwargs: Mapping[str, Any],
    ) -> None:
        """Adapt a supplied vLLM engine with an explicit one-completion recipe.

        The factory owns complete sampling configuration. The adapter adds only
        the per-request seed. The parser must locate spans without changing IDs;
        parsing/decoding a summary never rewrites the actor's native history.
        """
        if sampling_settings.get("n") != 1:
            raise ValueError("sampling_settings must explicitly request n=1")
        if "seed" in sampling_settings:
            raise ValueError("supply seeds per generation, not in sampling_settings")
        if {"prompts", "sampling_params"}.intersection(engine_generate_kwargs):
            raise ValueError("engine_generate_kwargs cannot override prompts or sampling_params")
        self.engine = engine
        self._policy_version_getter = policy_version_getter
        self.sampling_params_factory = sampling_params_factory
        self.sampling_settings = dict(sampling_settings)
        self.generation_parser = generation_parser
        self.engine_generate_kwargs = dict(engine_generate_kwargs)

    @property
    def policy_version(self) -> str:
        return self._policy_version_getter()

    def generate(self, prefix_token_ids: Sequence[int], *, seed: int) -> Generation:
        prefix = _tokens(prefix_token_ids, label="vLLM prompt")
        if not prefix or type(seed) is not int or seed < 0:
            raise ValueError("vLLM requires a nonempty prefix and an explicit nonnegative seed")
        version = self.policy_version
        if not isinstance(version, str) or not version:
            raise PolicyVersionError("vLLM requires an identified frozen policy version")
        params = self.sampling_params_factory(**self.sampling_settings, seed=seed)
        requests = self.engine.generate(
            prompts=[{"prompt_token_ids": list(prefix)}],
            sampling_params=params,
            **self.engine_generate_kwargs,
        )
        if self.policy_version != version:
            raise PolicyVersionError("vLLM policy changed during generation")
        if len(requests) != 1 or len(requests[0].outputs) != 1:
            raise BackendContractError("vLLM must return exactly one request and one completion")
        if tuple(requests[0].prompt_token_ids) != prefix:
            raise BackendContractError("vLLM returned a changed native-token prompt")
        completion = requests[0].outputs[0]
        ids = _tokens(completion.token_ids, label="vLLM completion")
        reason = completion.finish_reason
        if not isinstance(reason, str):
            raise BackendContractError("vLLM completion must have a final finish_reason")
        result = self.generation_parser(ids, reason)
        if (not isinstance(result, Generation) or result.token_ids != ids
                or result.finish_reason != reason):
            raise MalformedGenerationError("generation parser changed native IDs or finish_reason")
        return result


class TransformersEntropyProvider:
    def __init__(
        self,
        *,
        model: Any,
        policy_version_getter: Callable[[], str],
        device: Any,
        chunk_size: int,
        max_context_tokens: int,
        model_call_kwargs: Mapping[str, Any],
    ) -> None:
        """Teacher-force a supplied eval-mode causal LM using its KV cache.

        Chunking limits full-vocabulary logits to chunk_size positions per call.
        No vocabulary top-k approximation or context-window truncation is used.
        Token zero has no preceding causal context, so its entropy is NaN. All
        generated tokens must follow a nonempty prompt and have finite entries.
        The model must implement the standard causal-LM logits/KV-cache contract.
        """
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be an explicit positive integer")
        if type(max_context_tokens) is not int or max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be an explicit positive integer")
        reserved = {
            "input_ids", "inputs_embeds", "attention_mask", "position_ids", "cache_position",
            "past_key_values", "use_cache", "return_dict", "labels", "logits_to_keep",
            "num_logits_to_keep",
        }
        if reserved.intersection(model_call_kwargs):
            raise ValueError("model_call_kwargs overrides an entropy alignment argument")
        self.model = model
        self._policy_version_getter = policy_version_getter
        self.device = device
        self.chunk_size = chunk_size
        self.max_context_tokens = max_context_tokens
        self.model_call_kwargs = dict(model_call_kwargs)

    @property
    def policy_version(self) -> str:
        return self._policy_version_getter()

    def token_entropies(self, trajectory: Trajectory) -> Sequence[float]:
        if not trajectory.policy_version or self.policy_version != trajectory.policy_version:
            raise PolicyVersionError("entropy model must match the trajectory's old policy")
        if getattr(self.model, "training", True):
            raise BackendContractError("caller must put the entropy model in eval mode")
        ids = _tokens(trajectory.token_ids, label="entropy trajectory")
        if not ids:
            raise BackendContractError("entropy trajectory cannot be empty")
        if len(ids) != len(trajectory.generated_mask) or trajectory.generated_mask[0]:
            raise BackendContractError("entropy requires an aligned mask and non-generated first token")
        if any(type(flag) is not bool for flag in trajectory.generated_mask):
            raise BackendContractError("entropy generated mask must contain boolean values")
        if len(ids) > self.max_context_tokens:
            raise RolloutLengthError("entropy computation cannot truncate the trajectory context")
        # Optional dependency is imported only when explicitly invoked by caller.
        import torch

        values = [nan]
        cache = None
        with torch.inference_mode():
            for start in range(0, len(ids) - 1, self.chunk_size):
                stop = min(start + self.chunk_size, len(ids) - 1)
                inputs = torch.tensor([ids[start:stop]], dtype=torch.long, device=self.device)
                attention = torch.ones((1, stop), dtype=torch.long, device=self.device)
                output = self.model(
                    input_ids=inputs,
                    attention_mask=attention,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    **self.model_call_kwargs,
                )
                logits = output.logits
                if (logits.ndim != 3 or logits.shape[:2] != (1, stop - start)
                        or logits.shape[-1] == 0 or not logits.is_floating_point()):
                    raise BackendContractError("causal LM returned misaligned per-token logits")
                if not torch.isfinite(logits).all().item():
                    raise BackendContractError("causal LM returned nonfinite full-vocabulary logits")
                if logits.dtype in (torch.float16, torch.bfloat16):
                    logits = logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                if not torch.isfinite(entropy).all().item():
                    raise BackendContractError("whole-vocabulary entropy is nonfinite")
                values.extend(entropy[0].cpu().tolist())
                cache = output.past_key_values
                if stop < len(ids) - 1 and cache is None:
                    raise BackendContractError("causal LM did not return the required KV cache")
                if self.policy_version != trajectory.policy_version:
                    raise PolicyVersionError("entropy model policy changed during teacher forcing")
        return tuple(values)

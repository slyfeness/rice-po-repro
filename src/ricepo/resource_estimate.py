"""Static byte estimates for the single-process FP32 actor/reference and AdamW."""


def estimate_resources(*, actor_parameters, reference_parameters, retriever_parameters,
                       sequence_length, vocabulary_size, hidden_size, layers, kv_heads,
                       head_dim, compute_bytes, microbatch_size=1):
    inputs = dict(actor_parameters=actor_parameters, reference_parameters=reference_parameters,
                  retriever_parameters=retriever_parameters, sequence_length=sequence_length,
                  vocabulary_size=vocabulary_size, hidden_size=hidden_size, layers=layers,
                  kv_heads=kv_heads, head_dim=head_dim, compute_bytes=compute_bytes,
                  microbatch_size=microbatch_size)
    if any(type(value) is not int or value <= 0 for value in inputs.values()) or compute_bytes not in (2, 4):
        raise ValueError("positive integer dimensions and 2-byte or 4-byte compute precision are required")
    tokens = sequence_length * microbatch_size
    resident = {"actor_fp32_parameters": 4 * actor_parameters, "actor_fp32_gradients": 4 * actor_parameters,
                "adamw_first_and_second_fp32_moments": 8 * actor_parameters,
                "reference_fp32_parameters": 4 * reference_parameters}
    return {
        "kind": "static_formula_estimate", "measured": False, "unit": "bytes", "inputs": inputs,
        "actor_device_persistent": resident, "actor_device_persistent_sum": sum(resident.values()),
        "retriever_device_weights": retriever_parameters * compute_bytes,
        "ppo_policy_and_reference_logits_at_compute_dtype": 2 * tokens * vocabulary_size * compute_bytes,
        "ppo_policy_and_reference_logits_if_fp32": 2 * tokens * vocabulary_size * 4,
        "ppo_activation_proxy_one_hidden_tensor_per_layer": tokens * layers * hidden_size * compute_bytes,
        "generation_kv_cache": 2 * layers * tokens * kv_heads * head_dim * compute_bytes,
        "assumptions": [
            "All actor parameters are trainable FP32, reference is separate FP32, retriever uses compute dtype.",
            "Dense AdamW has two FP32 moment tensors per parameter after its first step.",
            "PPO uses full-sequence policy autograd and frozen reference logits with use_cache=False.",
            "Generation uses unshared, unquantized K and V at the declared length and batch size.",
        ],
        "unmodeled": ["autograd saved tensors, attention and MLP intermediates",
                      "full-vocabulary normalization and backward temporaries", "mixed-precision weight casts",
                      "optimizer implementation workspaces", "generation batches and cache allocation strategy",
                      "retriever forward/index memory", "allocator fragmentation and other processes"],
        "peak_bytes": None,
    }

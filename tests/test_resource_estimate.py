import pytest

from ricepo.resource_estimate import estimate_resources


def test_static_components_separate_two_devices_and_phase_specific_tensors():
    result = estimate_resources(actor_parameters=10, reference_parameters=10, retriever_parameters=20,
                                sequence_length=7, vocabulary_size=5, hidden_size=8, layers=2,
                                kv_heads=1, head_dim=4, compute_bytes=2)
    assert result["actor_device_persistent_sum"] == 200 and result["retriever_device_weights"] == 40
    assert result["generation_kv_cache"] == 2 * 2 * 7 * 1 * 4 * 2
    assert result["ppo_policy_and_reference_logits_if_fp32"] == 2 * 7 * 5 * 4
    assert result["ppo_activation_proxy_one_hidden_tensor_per_layer"] == 7 * 2 * 8 * 2
    assert result["peak_bytes"] is None and result["measured"] is False


def test_invalid_precision_rejected():
    with pytest.raises(ValueError):
        estimate_resources(actor_parameters=1, reference_parameters=1, retriever_parameters=1,
                           sequence_length=1, vocabulary_size=1, hidden_size=1, layers=1,
                           kv_heads=1, head_dim=1, compute_bytes=True)

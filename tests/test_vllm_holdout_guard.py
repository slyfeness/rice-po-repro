"""Do not silently replace a selected vLLM policy with the HF sampler."""
import pytest
from ricepo.holdout import _actor_runtime


def test_vllm_standalone_holdout_rejects_before_loading_assets():
    with pytest.raises(ValueError, match="standalone vLLM holdout is not implemented"):
        _actor_runtime(None, None, None, {},
                       {"sampling_backend": "vllm_ieee_two_stage_v1"}, lambda: "policy")

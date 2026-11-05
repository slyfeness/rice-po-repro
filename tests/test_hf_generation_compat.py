"""Regression coverage for the Transformers generation API compatibility boundary."""

from contextlib import nullcontext

import pytest

from ricepo.hf_runtime import HFNativeGenerator


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 2

    def decode(self, token_ids, **_kwargs):
        return "</summary>" if token_ids else ""


def test_hf_generation_uses_explicit_recipe_when_api_has_no_use_model_defaults_keyword():
    """A passed GenerationConfig remains the complete sampling law without the old keyword."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    prefix, seed = (210, 214, 214), 43
    captured = {}

    class Model:
        training = False

        def generate(self, *, input_ids, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[*prefix, 7]])

    generator = HFNativeGenerator(model=Model(), tokenizer=_Tokenizer(), device="cpu",
                                  policy_version_getter=lambda: "frozen-policy",
                                  max_new_tokens=8, max_context_tokens=64,
                                  prefilled_think=False, precision_context=nullcontext)
    generator.parser = lambda token_ids, reason: (token_ids, reason)
    assert generator.generate(prefix, seed=seed) == ((7,), "stop")
    assert "use_model_defaults" not in captured
    recipe = captured["generation_config"]
    assert (recipe.do_sample, recipe.temperature, recipe.top_p, recipe.top_k) == (True, 1.0, 1.0, 0)
    assert captured["logits_processor"].__class__.__name__ == "LogitsProcessorList"

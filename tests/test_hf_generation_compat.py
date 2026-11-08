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


def test_eos_after_think_with_an_unclosed_summary_is_not_marked_stop():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    prefix = (210, 214, 214)

    class Tokenizer:
        bos_token_id = None
        eos_token_id = 2
        pad_token_id = 2

        def decode(self, token_ids, **_kwargs):
            return "</think><summary><summary>" if token_ids else ""

    class Model:
        training = False

        def generate(self, *, input_ids, **_kwargs):
            return torch.tensor([[*prefix, 7, 2]])

    generator = HFNativeGenerator(model=Model(), tokenizer=Tokenizer(), device="cpu",
                                  policy_version_getter=lambda: "frozen-policy",
                                  max_new_tokens=8, max_context_tokens=64,
                                  prefilled_think=True, precision_context=nullcontext,
                                  summary_mode="summary_end_right_inclusive_v1")
    class Parser:
        summary_mode = "summary_end_right_inclusive_v1"

        def __call__(self, _ids, reason):
            return reason

    generator.parser = Parser()
    assert generator.generate(prefix, seed=43) == "eos"


def test_saved_native_replay_never_calls_model_generation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    class Model:
        training=False
        def generate(self,**kwargs):raise AssertionError("must reuse paid IDs")
    generator=HFNativeGenerator(model=Model(),tokenizer=_Tokenizer(),device="cpu",
        policy_version_getter=lambda:"frozen",max_new_tokens=8,max_context_tokens=64,
        prefilled_think=False,precision_context=nullcontext)
    generator.completion_replay=lambda prefix,seed,version:dict(token_ids=(7,2),finish_reason="eos",source={"event_sha256":"a"*64})
    generator.parser=lambda ids,reason:(ids,reason)
    assert generator.generate((210,214),seed=43)==((7,2),"stop")

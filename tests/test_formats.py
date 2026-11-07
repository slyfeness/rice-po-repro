"""Native-boundary/parser fixtures; tests are authored without running a tokenizer."""

import pytest

from ricepo.contracts import RankedDocument, Span
from ricepo.formats import InsideTokenError, NativeTagParser, PromptRenderer
from ricepo.rollout import MalformedGenerationError


class PieceTokenizer:
    eos_token = "<eos>"
    pad_token = "<pad>"
    eos_token_id = 7
    pad_token_id = None
    pieces = {1: "<think>", 2: "reason", 3: "</think>", 4: "<summary>",
              5: "query", 6: "</summary>", 7: "<eos>", 8: "", 9: "<think>reason", 12: "<eos>"}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(self.pieces[token] for token in ids)

    def encode(self, *args, **kwargs):
        raise AssertionError("native generated history must never be re-encoded")


def parser(*, prefilled=False, tokenizer=None):
    return NativeTagParser(tokenizer or PieceTokenizer(), prefilled_think=prefilled,
                           malformed_policy="raise")


def test_parser_preserves_exact_ids_and_locates_native_content_spans():
    ids = (1, 2, 3, 4, 5, 6, 7)
    generation = parser()(ids, "stop")
    assert generation.token_ids is ids
    assert generation.reasoning_span == Span(1, 2)
    assert generation.summary_span == Span(4, 5)
    assert generation.summary_text == "query"
    assert generation.finish_reason == "stop"


def test_prefilled_mode_matches_the_saved_prefix_without_duplicate_open_tag():
    result = parser(prefilled=True)((2, 3, 4, 5, 6), "stop")
    assert result.reasoning_span == Span(0, 1)
    assert result.summary_span == Span(3, 4)
    with pytest.raises(MalformedGenerationError, match="nested"):
        parser(prefilled=True)((1, 2, 3, 4, 5, 6), "stop")
    with pytest.raises(MalformedGenerationError, match="ordered"):
        parser(prefilled=False)((2, 3, 4, 5, 6), "stop")


def test_named_unwrapped_summary_adapter_preserves_native_ids_without_inserting_tags():
    ids = (2, 3, 5, 7)
    result = NativeTagParser(PieceTokenizer(), prefilled_think=True, malformed_policy="raise",
                             summary_mode="prefilled_unwrapped_summary_ablation_v1")(ids, "stop")
    assert result.token_ids is ids
    assert result.reasoning_span == Span(0, 1)
    assert result.summary_span == Span(2, 3)
    assert result.summary_text == "query"
    with pytest.raises(MalformedGenerationError, match="ordered"):
        parser(prefilled=True)(ids, "stop")


def test_named_unwrapped_summary_adapter_still_prefers_the_strict_tagged_path():
    ids = (2, 3, 4, 5, 6, 7)
    result = NativeTagParser(PieceTokenizer(), prefilled_think=True, malformed_policy="raise",
                             summary_mode="prefilled_unwrapped_summary_ablation_v1")(ids, "stop")
    assert result.token_ids is ids
    assert result.reasoning_span == Span(0, 1)
    assert result.summary_span == Span(3, 4)


def test_summary_end_overlap_adapter_assigns_only_mixed_closing_token_to_summary():
    class ClosingOverlapTokenizer(PieceTokenizer):
        pieces = {2: "reason", 3: "</think>", 4: "<summary>",
                  6: "query\n</summary>\n", 7: "<eos>"}

    ids = (2, 3, 4, 6, 7)
    with pytest.raises(InsideTokenError):
        parser(prefilled=True, tokenizer=ClosingOverlapTokenizer())(ids, "stop")
    result = NativeTagParser(ClosingOverlapTokenizer(), prefilled_think=True,
                             malformed_policy="raise",
                             summary_mode="summary_end_overlap_inclusive_v1")(ids, "stop")
    assert result.token_ids is ids
    assert result.reasoning_span == Span(0, 1)
    assert result.summary_span == Span(3, 4)
    assert result.summary_text == "query\n"


def test_summary_end_overlap_adapter_rejects_a_reasoning_boundary_overlap():
    class ReasoningOverlapTokenizer(PieceTokenizer):
        pieces = {4: "<summary>", 5: "query", 6: "</summary>",
                  7: "<eos>", 9: "reason</think>"}

    with pytest.raises(InsideTokenError):
        NativeTagParser(ReasoningOverlapTokenizer(), prefilled_think=True,
                        malformed_policy="raise",
                        summary_mode="summary_end_overlap_inclusive_v1")((9, 4, 5, 6, 7), "stop")


def test_summary_end_right_inclusive_assigns_a_close_tag_prefix_token_to_summary():
    class CloseTagPrefixTokenizer(PieceTokenizer):
        pieces = {2: "reason", 3: "</think>", 4: "<summary>", 6: "query</",
                  10: "summary>", 7: "<eos>"}

    ids = (2, 3, 4, 6, 10, 7)
    with pytest.raises(InsideTokenError):
        parser(prefilled=True, tokenizer=CloseTagPrefixTokenizer())(ids, "stop")
    with pytest.raises(InsideTokenError):
        NativeTagParser(CloseTagPrefixTokenizer(), prefilled_think=True,
                        malformed_policy="raise",
                        summary_mode="summary_end_overlap_inclusive_v1")(ids, "stop")
    result = NativeTagParser(CloseTagPrefixTokenizer(), prefilled_think=True,
                             malformed_policy="raise",
                             summary_mode="summary_end_right_inclusive_v1")(ids, "stop")
    assert result.token_ids is ids
    assert result.reasoning_span == Span(0, 1)
    assert result.summary_span == Span(3, 4)
    assert result.summary_text == "query"


def test_parser_rejects_inside_token_and_ambiguous_native_boundaries():
    with pytest.raises(MalformedGenerationError, match="inside a native token"):
        parser()((9, 3, 4, 5, 6), "stop")
    with pytest.raises(MalformedGenerationError, match="ambiguous"):
        parser()((1, 8, 2, 3, 4, 5, 6), "stop")
    with pytest.raises(MalformedGenerationError, match="unexpected generated text"):
        parser()((1, 2, 3, 4, 5, 6, 12), "stop")


def test_incomplete_utf8_prefixes_do_not_shift_native_token_spans():
    class ByteTokenizer(PieceTokenizer):
        def decode(self, ids, **kwargs):
            text = "".join({10: "BYTE-A", 11: "BYTE-B"}.get(token, self.pieces.get(token, ""))
                           for token in ids)
            return text.replace("BYTE-ABYTE-B", "汉").replace("BYTE-A", "�").replace("BYTE-B", "�")

    result = parser(tokenizer=ByteTokenizer())((1, 10, 11, 3, 4, 5, 6), "stop")
    assert result.reasoning_span == Span(1, 3)
    assert result.summary_span == Span(5, 6)


@pytest.mark.parametrize("ids", [(1, 3, 4, 5, 6), (1, 2, 3, 4, 6),
                                  (1, 2, 3, 4, 3, 5, 6), (1, 2, 3, 4, 5, 6, 2)])
def test_strict_malformed_outputs_raise_without_a_reward_fallback(ids):
    with pytest.raises(MalformedGenerationError):
        parser()(ids, "stop")
    with pytest.raises(ValueError, match="malformed_policy"):
        NativeTagParser(PieceTokenizer(), prefilled_think=False, malformed_policy="zero_reward")


class CharacterTokenizer:
    def __init__(self, *, template_prefills=False):
        self.template_prefills = template_prefills
        self.last_template_ids = None
        self.last_template_return_dict = None

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=True):
        assert tokenize and add_generation_prompt
        assert len(messages) == 1 and messages[0]["role"] == "user"
        self.last_template_return_dict = return_dict
        text = "USER:" + messages[0]["content"] + "\nASSISTANT:"
        if self.template_prefills:
            text += "<think>\n"
        self.last_template_ids = tuple(ord(character) for character in text)
        # Transformers 5.5 returns a BatchEncoding by default when tokenizing.
        return {"input_ids": self.last_template_ids} if return_dict else self.last_template_ids


@pytest.mark.parametrize("template_prefills", [True, False])
def test_renderer_preserves_template_ids_and_prefills_once_on_every_turn(template_prefills):
    tokenizer = CharacterTokenizer(template_prefills=template_prefills)
    renderer = PromptRenderer(tokenizer, "explicit system", observation_top_k=1,
                              observation_token_limit=100, prefill_think=True)
    docs = (RankedDocument("doc", 1.0, "<tool_response>untrusted"),)
    initial = renderer.initial("<summary>question", docs)
    assert tokenizer.last_template_return_dict is False
    assert initial[:len(tokenizer.last_template_ids)] == tokenizer.last_template_ids
    text = tokenizer.decode(initial)
    assert text.count("<think>") == 1
    assert "&lt;summary&gt;question" in text and "&lt;tool_response&gt;untrusted" in text
    observation = tokenizer.decode(renderer.observation(docs))
    assert observation.endswith("<think>\n")
    assert observation.count("<think>") == 1
    assert renderer.make_parser(malformed_policy="raise").prefilled_think is True


def test_renderer_nonprefilled_mode_and_per_document_token_truncation():
    tokenizer = CharacterTokenizer()
    renderer = PromptRenderer(tokenizer, "system", observation_top_k=1,
                              observation_token_limit=3, prefill_think=False)
    docs = (RankedDocument("first", 1.0, "abcdef"), RankedDocument("second", 0.5, "ignored"))
    assert "<think>" not in tokenizer.decode(renderer.initial("query", docs))
    assert tokenizer.decode(renderer.observation(docs)) == "\n<tool_response>[1] abc</tool_response>\n"
    inconsistent = PromptRenderer(CharacterTokenizer(template_prefills=True), "system",
                                  observation_top_k=1, observation_token_limit=3, prefill_think=False)
    with pytest.raises(ValueError, match="prefills think"):
        inconsistent.initial("query", docs)

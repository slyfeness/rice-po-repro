"""Native-token segmentation and explicit prompt rendering conventions.

Decode is used only to locate spans and create the retrieval query. Generated
tokens are NEVER reconstructed with encode(decode(ids)). Ambiguous boundaries
raise an error instead of silently shifting policy-loss spans.
"""

import html
import re
from .contracts import Generation, Span
from .rollout import MalformedGenerationError


class EmptySpanError(MalformedGenerationError):
    code = "empty_span"


class InsideTokenError(MalformedGenerationError):
    code = "tag_inside_token"


class AmbiguousBoundaryError(MalformedGenerationError):
    code = "ambiguous_tokenizer_boundary"


class NativeTagParser:
    """Strict reference parser: prefix decoding may take quadratic work.

    Only uniquely mapped native boundaries are accepted. A tokenizer-specific
    faster parser can be injected into the generator after separate validation.
    """
    def __init__(self, tokenizer, *, prefilled_think: bool, malformed_policy: str):
        if type(prefilled_think) is not bool or malformed_policy != "raise":
            raise ValueError("choose an explicit prefilled_think boolean and malformed_policy='raise'")
        self.tokenizer = tokenizer
        self.prefilled_think = prefilled_think

    def __call__(self, token_ids: tuple[int, ...], finish_reason: str) -> Generation:
        if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
            raise MalformedGenerationError("expected nonempty native integer token IDs")
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
        terminal_ids = {token for token in (getattr(self.tokenizer, "eos_token_id", None),
                                             getattr(self.tokenizer, "pad_token_id", None))
                        if type(token) is int and token >= 0}
        content_end = len(token_ids)
        while content_end and token_ids[content_end - 1] in terminal_ids:
            content_end -= 1
        text = decode(token_ids[:content_end])
        pattern = (r"(.*?)</think>\s*<summary>(.*?)</summary>" if self.prefilled_think
                   else r"\s*<think>(.*?)</think>\s*<summary>(.*?)</summary>")
        match = re.match(pattern, text, flags=re.DOTALL)
        if match is None:
            raise MalformedGenerationError("expected ordered think and summary tags")
        tail = text[match.end():]
        # Only whitespace and an actual suffix of declared EOS/pad IDs may
        # follow the closing tag. Ordinary tokens spelling '<eos>' do not count.
        if tail.strip():
            raise MalformedGenerationError("unexpected generated text after summary")
        if not match.group(1).strip():
            raise EmptySpanError("strict parsing requires nonempty reasoning")
        if not match.group(2).strip():
            raise EmptySpanError("strict parsing requires nonempty summary")
        if any(tag in match.group(1) or tag in match.group(2)
               for tag in ("<think>", "</think>", "<summary>", "</summary>",
                           "<tool_response>", "</tool_response>")):
            raise MalformedGenerationError("nested or duplicate output tags")
        targets = {match.start(1), match.end(1), match.start(2), match.end(2)}
        boundaries = {0: [0]}
        # Some tokenizers temporarily decode incomplete UTF-8 bytes as U+FFFD.
        # Accept only prefixes that are exact prefixes of the complete text.
        for end in range(1, content_end + 1):
            prefix = decode(token_ids[:end])
            if len(prefix) in targets and text.startswith(prefix):
                boundaries.setdefault(len(prefix), []).append(end)
        if not targets.issubset(boundaries):
            raise InsideTokenError("tag/content boundary lies inside a native token")
        if any(len(boundaries[target]) != 1 for target in targets):
            raise AmbiguousBoundaryError("tag/content boundary has ambiguous native token offsets")
        reasoning = Span(boundaries[match.start(1)][0], boundaries[match.end(1)][0])
        summary = Span(boundaries[match.start(2)][0], boundaries[match.end(2)][0])
        return Generation(token_ids, reasoning, summary, match.group(2), finish_reason)


class PromptRenderer:
    """Figure 5 text + escaped query/documents. Layout is a documented choice.

    No qrels, gold IDs, reference reasoning, reward, or advantage is accepted.
    Tool snippets use actor-tokenizer truncation, separately from retrieval rank.
    """
    def __init__(self, tokenizer, system_prompt: str, *, observation_top_k: int,
                 observation_token_limit: int, prefill_think: bool):
        if (not isinstance(system_prompt, str) or not system_prompt
                or type(observation_top_k) is not int or observation_top_k <= 0
                or type(observation_token_limit) is not int or observation_token_limit <= 0
                or type(prefill_think) is not bool):
            raise ValueError("explicit prompt and positive observation limits required")
        self.tokenizer, self.system_prompt = tokenizer, system_prompt
        self.top_k, self.token_limit = observation_top_k, observation_token_limit
        self.prefill_think = prefill_think

    def make_parser(self, *, malformed_policy: str) -> NativeTagParser:
        """Bind strict parsing to exactly the prefill mode used at every branch prefix."""
        return NativeTagParser(self.tokenizer, prefilled_think=self.prefill_think,
                               malformed_policy=malformed_policy)

    def _documents(self, documents) -> str:
        records = []
        for rank, document in enumerate(documents[:self.top_k], 1):
            ids = self.tokenizer.encode(document.text, add_special_tokens=False)[:self.token_limit]
            snippet = self.tokenizer.decode(ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
            records.append(f"[{rank}] {html.escape(snippet)}")
        return "\n".join(records)

    def initial(self, query: str, documents) -> tuple[int, ...]:
        message = (self.system_prompt + "\n<query>" + html.escape(query) + "</query>\n<information>"
                   + self._documents(documents) + "</information>")
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        if (not isinstance(rendered, (list, tuple))
                or any(type(token) is not int or token < 0 for token in rendered)):
            raise ValueError("chat template must return native integer token IDs")
        ids = tuple(rendered)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        already_prefilled = decoded.rstrip().endswith("<think>")
        if already_prefilled and not self.prefill_think:
            raise ValueError("chat template prefills think but the explicit rendering mode disables it")
        if self.prefill_think and not already_prefilled:
            ids += tuple(self.tokenizer.encode("<think>\n", add_special_tokens=False))
        return ids

    def observation(self, documents) -> tuple[int, ...]:
        text = "\n<tool_response>" + self._documents(documents) + "</tool_response>\n"
        if self.prefill_think:
            text += "<think>\n"
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

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
    """Native token parser; strict tagged parsing is the reference mode.

    Only uniquely mapped native boundaries are accepted. A tokenizer-specific
    faster parser can be injected into the generator after separate validation.
    """
    STRICT_TAGGED = "strict_tagged"
    PREFILLED_UNWRAPPED_SUMMARY = "prefilled_unwrapped_summary_ablation_v1"
    SUMMARY_END_OVERLAP_INCLUSIVE_V1 = "summary_end_overlap_inclusive_v1"

    def __init__(self, tokenizer, *, prefilled_think: bool, malformed_policy: str,
                 summary_mode: str = STRICT_TAGGED):
        if type(prefilled_think) is not bool or malformed_policy != "raise":
            raise ValueError("choose an explicit prefilled_think boolean and malformed_policy='raise'")
        if summary_mode not in {self.STRICT_TAGGED, self.PREFILLED_UNWRAPPED_SUMMARY,
                                self.SUMMARY_END_OVERLAP_INCLUSIVE_V1}:
            raise ValueError("unsupported native summary parsing mode")
        if summary_mode in {self.PREFILLED_UNWRAPPED_SUMMARY,
                            self.SUMMARY_END_OVERLAP_INCLUSIVE_V1} and not prefilled_think:
            raise ValueError("unwrapped-summary mode requires an explicitly prefilled think prefix")
        self.tokenizer = tokenizer
        self.prefilled_think = prefilled_think
        self.summary_mode = summary_mode

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
        unwrapped = self.summary_mode in {self.PREFILLED_UNWRAPPED_SUMMARY,
                                          self.SUMMARY_END_OVERLAP_INCLUSIVE_V1}
        has_summary_tag = "<summary>" in text or "</summary>" in text
        if unwrapped and not has_summary_tag:
            # This is a separately named non-paper format adapter.  It maps
            # the model's EOS-terminated text after a native </think> token
            # boundary to the summary span; it never inserts, re-encodes, or
            # otherwise fabricates tag tokens.
            pattern = r"(.*?)</think>\s*(.*?)\Z"
        else:
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
        prefixes = {0: ""}
        # Some tokenizers temporarily decode incomplete UTF-8 bytes as U+FFFD.
        # Accept only prefixes that are exact prefixes of the complete text.
        for end in range(1, content_end + 1):
            prefix = decode(token_ids[:end])
            prefixes[end] = prefix
            if len(prefix) in targets and text.startswith(prefix):
                boundaries.setdefault(len(prefix), []).append(end)
        missing = targets - boundaries.keys()
        summary_end = match.end(2)
        if (missing == {summary_end}
                and self.summary_mode == self.SUMMARY_END_OVERLAP_INCLUSIVE_V1):
            candidates = []
            for end in range(1, content_end + 1):
                left, right = prefixes[end - 1], prefixes[end]
                if not (text.startswith(left) and text.startswith(right)
                        and len(left) < summary_end < len(right)):
                    continue
                token_text = right[len(left):]
                before, after = token_text[:summary_end - len(left)], token_text[summary_end - len(left):]
                leading, tag, trailing = after.partition("</summary>")
                if before.strip() and tag and not leading.strip() and not trailing.strip():
                    candidates.append(end)
            if len(candidates) == 1:
                boundaries[summary_end] = candidates
                missing = set()
            elif len(candidates) > 1:
                raise AmbiguousBoundaryError("summary end has multiple native overlap interpretations")
        if missing:
            raise InsideTokenError("tag/content boundary lies inside a native token")
        if any(len(boundaries[target]) != 1 for target in targets):
            raise AmbiguousBoundaryError("tag/content boundary has ambiguous native token offsets")
        reasoning = Span(boundaries[match.start(1)][0], boundaries[match.end(1)][0])
        summary = Span(boundaries[match.start(2)][0], boundaries[match.end(2)][0])
        return Generation(token_ids, reasoning, summary, match.group(2), finish_reason)


def native_summary_mode(deployment_value: str) -> str:
    """Translate an auditable deployment selector to a parser implementation.

    The strict tagged mode is the independent reference.  The alternate name
    is deliberately verbose so an ablation cannot be mistaken for it.
    """
    modes = {
        "strict_native_tags": NativeTagParser.STRICT_TAGGED,
        "prefilled_think_unwrapped_summary_ablation_v1": NativeTagParser.PREFILLED_UNWRAPPED_SUMMARY,
        "summary_end_overlap_inclusive_v1": NativeTagParser.SUMMARY_END_OVERLAP_INCLUSIVE_V1,
    }
    try:
        return modes[deployment_value]
    except (KeyError, TypeError) as error:
        raise ValueError("unsupported output_parser_mode deployment value") from error


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

    def make_parser(self, *, malformed_policy: str,
                    summary_mode: str = NativeTagParser.STRICT_TAGGED) -> NativeTagParser:
        """Bind strict parsing to exactly the prefill mode used at every branch prefix."""
        return NativeTagParser(self.tokenizer, prefilled_think=self.prefill_think,
                               malformed_policy=malformed_policy, summary_mode=summary_mode)

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

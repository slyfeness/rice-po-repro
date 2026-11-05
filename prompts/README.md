# RICE-PO agent prompt

[`ricepo_system.txt`](ricepo_system.txt) is a transcription of the **body of
Figure 5**, Appendix A.2 of [RICE-PO, arXiv:2605.26352v3](https://arxiv.org/html/2605.26352v3#A1.F5).
The source was checked on 2026-11-04. The figure is an inline SVG whose
`foreignObject` contains selectable text, including tags and word limits;
this transcription did not rely on OCR or on the Diver repository.

## Attribution and transcription

Source: **Mingchen Li, Hansi Zeng, Zhuo Qian, Jiatan Huang, Sunjae Kwon,
Hamed Zamani, and Hong Yu**, “RICE-PO: Turning Retrieval Interactions into
Credit Signals for Reasoning Agents,” arXiv:2605.26352v3 (6 August 2026),
Figure 5. The [paper's license link](https://arxiv.org/abs/2605.26352v3)
identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The transcribed prompt remains under that license; it is not relicensed by
the repository's software license. No author endorsement is implied.

The figure heading and caption, typography, colors, and SVG markup are omitted.
Paragraphs and line breaks are normalized for plain text. HTML entities are
decoded, so `&lt;summary&gt;` becomes `<summary>`. Wording, punctuation, case,
and the en dashes in the numerical ranges are retained. The local UTF-8 file,
including its whitespace and final newline, is an implementation artifact;
the paper does not publish an original byte-for-byte prompt file or chat
serialization.

Static verification on 2026-11-04 compared the decoded Figure 5 body with the
local file after whitespace normalization: all wording matched. The local
file's SHA-256 is
`e2241e76e95dc773dc5250afc68ff352b655ae2e631a0db572855a2f7cebe04f`.
This is a text comparison, not a model or prompt-execution result.

## Integration contract

- The initial input uses `<query> ... </query>` and
  `<information> ... </information>`; later observations use
  `<tool_response> ... </tool_response>`.
- Each generated step contains a reasoning span followed by a summary span.
  The prompt asks for no generated text or placeholder after `</summary>`.
- The printed limits are **200 words** for reasoning and **0–500 words** for
  the summary, plus 4–8 keywords/entities and at least one exclusion or
  disambiguation phrase. These are prompt instructions, not published
  tokenizer limits. A token generation limit must be recorded separately;
  truncating at 200 tokens is not equivalent to the printed instruction.
- The filename identifies the instruction body, not an author-specified chat
  role. The reference `formats.PromptRenderer` places the instruction and
  query/documents together in a user message, using the pinned tokenizer's
  chat template; this is an explicit rendering convention. The
  figure does not determine the model's chat roles, special tokens,
  `add_generation_prompt`, or any prefilled `<think>`. Those choices must
  preserve the pinned tokenizer's native token IDs, and be recorded alongside
  the prompt hash. A parser must not regenerate the history by decoding and
  re-tokenizing it.
- A nonempty-summary parser or abort-on-malformed-output rule is an explicit
  engineering convention, because the printed lower word limit is zero and
  the paper does not define a failure reward. Never silently introduce a
  format penalty or repair the output into a different training sequence.
- Figure 5 and §3 say that the summary is the next retrieval query, whereas
  [§4.2.2](https://arxiv.org/html/2605.26352v3#S4.SS2.SSS2) says to use the summary
  together with the input query. The query composer therefore records its
  chosen composition and separator explicitly. This prompt is kept intact;
  it is not silently rewritten to resolve that discrepancy.

The original Diver expansion prompt in `third_party/reference/diver/` is
reference material for an external dependency and must not replace this
RICE-PO prompt. Having a transcribed prompt does not establish that model
loading, parsing, rollout, or training has been executed or validated.

## Compact format diagnostic (non-paper)

[`ricepo_prefilled_think_compact_format_v1.txt`](ricepo_prefilled_think_compact_format_v1.txt)
is an explicitly non-paper functional-diagnostic prompt. It removes the
reference prompt's diversity and length instructions and concentrates on one
unambiguous Think/ Summary closure. It may only be selected through
[`configs/format_prompt_diagnostic.json`](../configs/format_prompt_diagnostic.json),
whose profile is `ablation`. Results from it cannot be reported as results of
the Figure 5 reference prompt or the paper configuration.

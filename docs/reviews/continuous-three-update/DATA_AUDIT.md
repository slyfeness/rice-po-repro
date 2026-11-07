# Frozen data and retrieval identity

This phase reuses the existing Biology bundle: 128 training, 32 development and 103 held-out test rows. No split, query, prompt, qrels, corpus, retriever, embedding file or exclusion policy was rebuilt or selected from development performance. Test identities remain bound for separation checks; official test evaluation was not run. Source sample count agreement remains different from author-identical sample identity.

| Asset | SHA-256 |
|---|---|
| Bundle receipt | `56b57c5dc5a22fe521eff09380ad02248372401b5be57760ea540a82ed32b303` |
| Train manifest | `818fce2b0d9e41b57425b329008173bdbc60320c43972b515a1c377c940f65b2` |
| Development manifest | `5639686d224963c22e17981135453495e508646a7c270752c804fdc064cc1b71` |
| Test manifest | `7512f5c961f1c4f61d59f2dd0e0852f31f4a558ee383bff979eabbcb921d2494` |
| Prefilled compatibility prompt | `e7b28d3b6c7aebf97018936e2d2a0750a0cd447d61cb8b4be8935beef57b3423` |

The continuation admission checks the parent's completed checkpoint/source and matching data, model, tokenizer, index and behavior binding. Both methods use the same ordered global query schedule: four queries at update one, then four at each of global updates two and three. Sampling comes from the frozen training manifest with root seed 42 and global update indices, not from observed rewards. Public per-update schedule hashes and pseudonymous query identities are in `EVIDENCE.json`. This is a four-query optimizer batch, not a 128-query paper batch. The existence of 128 training rows does not mean all were trained on.

Actual actor assets retain DeepSeek revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`; Diver retains revision `ff7f8bc8d1827734dcf2bbfab99f065be618069a`. FP32 actor and BF16 Diver match the previously qualified independent setup. No new upstream verification claim is inferred from matching local fingerprints. Fixed Diver+BM25, `summary_only` and the full existing Biology corpus remain unchanged. The continuation/source bindings are separate from the original asset identities.

Existing 96 development trajectories were checked against all32 expected identities, not just row counts. Original D0 and legal per-round rankings were rescored against frozen qrels; stored NDCG agreed. The three paired update-one score disagreements also received positive-ID/corpus checks. Full questions, Summaries, qrels rows and document excerpts remain private. Descriptive results and evidence hashes appear in `RETRIEVAL_AUDIT.md`; no query was removed because its text/domain content or reward was inconvenient.

Final evaluation includes every one of the 32 frozen queries. Illegal terminations contribute registered task utility zero, while their actual last-valid NDCG remains recorded. Conditional legal-completion NDCG always has its own denominator and matched D0. No best-of-history or gold-guided sample selection is used.

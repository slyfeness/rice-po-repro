# Retrieval assets and reference comparison

## Vector bundle

Set `corpus_jsonl` to the source export and both output paths to files in a new sibling version directory, for example `domain/enc-v2/vectors.npy` and `domain/enc-v2/vector_document_ids.json`. Keep all three under storage. The encoding stage writes the vector matrix, the ID list and `vectors.receipt.json` into a private sibling directory. It fsyncs the complete files and publishes that directory while holding a sibling exclusive advisory lock. Every project publisher must use that lock; the filesystem must support coherent flock and atomic same-filesystem directory rename. Existing targets, including empty directories, are rejected under the lock. A killed writer can leave an unpublished `.pending-*` directory. Retrying may target the same absent final directory. Existing final directories are immutable.

Receipt version 2 binds the following evidence.

- SHA-256 of every regular payload file in the standalone local HF model directory, including sharded weights, shard indices, config, tokenizer files and nested templates. HF management directories `.git` and `.cache` are excluded. An index must point to hashed weight files. The encoder loader reads the manifest before and after loading the local tokenizer and frozen model. Unresolved or mismatched weights stop the load.
- Model and tokenizer class/module identity, package versions, encoder implementation digest, runtime attention implementation, actual dtype, special token IDs, query/document prefixes, padding and truncation side, length, pooling and normalization. Actual document batching is recorded. Output is a float32 matrix.
- SHA-256 of the exact corpus bytes, ordered JSON-escaped ID/text rows, ordered ID list, ID-file bytes and matrix-file bytes. Dimensions come from the loaded encoder. The consumer checks the exact ID sequence against the projected corpus and checks that every matrix row is finite and L2-normalized.

A source revision records the intended upstream retriever. Local payload matching checks whether the same bytes and implementation were used on both sides of the bundle. The receipt labels upstream payload verification `not_verified`. Keep the authorized file store immutable while reading bundles. Independent upstream manifests or output captures provide separate evidence. Changing the local encoding implementation or weights, even under the same name, requires a new version directory.

## Actual ranking recipe

Dense retrieval computes float32 cosine scores against all domain documents. Sparse retrieval uses the default Pyserini Lucene analyzer and a Gensim LuceneBM25Model with k1 = 0.9 and b = 0.4. Both the document and query bags are BM25-transformed. The sparse matrix index disables query and document normalization. Both scorers see the same input query.

Each score component is min/max normalized on the configured domain. A constant component contributes zero. The two normalized components have equal weight. The fixed per-query exclusions are applied at the configured normalization stage. Surviving documents are ordered by fused score descending and document ID ascending before the explicit cutoff. Per-result documents carry fused, raw dense/sparse and normalized dense/sparse scores. The recipe records candidate scope, cutoff, implementation digests, package versions and a content digest of analyzed document terms in corpus order. Raw fractional pytrec_eval NDCG@10 uses the returned rank order.

The pinned upstream Diver runner also offers a vLLM document profile with a `Represent this text:` prefix, a token/decode truncation pass and stable insertion-order ties in its component score helper. This local profile uses raw documents, HF tokenization and an explicit lexicographic hybrid tie rule. The upstream component helpers can cap candidates at 1,000. Record which upstream path, profile, candidate depth and normalization procedure produced each expectation. Imported cutoffs or different document profiles can legitimately produce differences. [Pinned upstream retrievers.py](../third_party/reference/diver/Retriever/retrievers.py)

## Frozen-case comparison

The following are interfaces for a separately authorized Slurm job. No upstream capture, compatibility check or ranking comparison has been run. Keep cases, outputs and reports on storage. Prepare a **small dedicated corpus** and publish its full vector bundle first. The collector builds the same full-domain sparse index and stores all document and query vectors in the report.

1. Supply a case JSON with `schema_version: 1`, `domain`, the SHA-256 of the corpus export in `corpus_sha256`, its full ordered `document_ids` and a nonempty `queries` list. Each query has a unique `id`, `original_query`, `summary` (null for original-query retrieval), `excluded_ids` and a `qrels` object mapping document IDs to nonnegative integer judgments. A non-null summary uses the explicit composition recipe from the config. Qrels are consumed by the reward scorer.
2. Collect observations from the local assets using the command below. Use an explicitly enabled research config and the matching deployed domain. The artifact includes the canonical JSON digest of the case. It stores the rendered inputs, padded IDs and masks in encoding batch order, actual matrix and query vectors, every raw/normalized component score, eligible fused scores, ranked IDs, and fractional NDCG@10.
3. Supply a **separately captured** JSON from the pinned upstream run. Use the same case inputs including batch/row order. Required header fields are `schema_version: 1`, `role: "upstream_expected"`, `case_sha256`, `source` and `layers`. The source object requires `repository` (`https://github.com/AQ-MedAI/Diver`), `revision` (`cfdf09ad863540f78a655880ef133c4cb28b3676`), `retriever_revision` (`ff7f8bc8d1827734dcf2bbfab99f065be618069a`), `producer`, the actual upstream `command` and `encoding_profile`. Include exact package/environment details or archive identifiers as additional source fields when available. Pin this file's SHA-256 independently before comparison. The checker verifies its bytes and declared revisions. Source authenticity is the responsibility of the evidence provider.
4. Compare the captures. The command writes a per-layer report with counts, maximum absolute error for numeric comparisons, and the first differing paths. It exits with status 1 when any layer differs or lacks upstream evidence. Missing layers remain visible so a component-only upstream capture can still be used. A complete match applies to this case, profile and supplied tolerances.

```sh
python -m ricepo.reference_compare collect --config "$CONFIG" --deployment "$DEPLOYMENT" \
    --case "$CASE" --output "$LOCAL_OBSERVATION"
python -m ricepo.reference_compare compare --observed "$LOCAL_OBSERVATION" \
    --expected "$EXTERNAL_CAPTURE" --expected-sha256 "$PINNED_CAPTURE_SHA256" --output "$REPORT"
```

The Layer names and JSON shapes are shared by both files.

| Layer | Value inside `layers` | Comparison |
|---|---|---|
| `rendered_inputs`, `token_ids`, `attention_masks` | Object with `documents` and `queries` maps, keyed by document ID or query ID. Values are rendered strings or integer arrays. | Exact, including padding. |
| `document_vectors`, `query_vectors` | Document ID or query ID to a numeric vector. | Numeric. |
| `dense_raw`, `sparse_raw`, `dense_normalized`, `sparse_normalized`, `fused` | Query ID to a map of document ID to score. Raw maps cover all rows. Normalized and fused maps reflect the declared exclusions. | Numeric, exact key sets. |
| `ranked_ids` | Query ID to ordered list of returned IDs. | Exact, including ties and cutoff. |
| `ndcg_cut_10` | Query ID to raw fractional score. | Numeric. |

For every numeric layer supplied, set its own `tolerances[layer]` to `{"atol": ..., "rtol": ...}` using independently chosen nonnegative finite bounds. The criterion is `abs(local - upstream) <= atol + rtol * abs(upstream)`. Omit unavailable upstream layers explicitly by leaving them out of the `layers` object. No expected measurements or default tolerances are shipped.

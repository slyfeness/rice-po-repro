"""Fixed-retriever primitives; unspecified RICE composition/fusion is explicit."""

from dataclasses import dataclass, replace
from itertools import chain
import math
from typing import Callable, Mapping, Sequence

from .contracts import RankedDocument, RetrievalResult
from .asset_identity import implementation_class, local_model_identity, package_versions, source_sha256
from .artifacts import require_cluster_storage, require_slurm
from .data import content_sha256


@dataclass(frozen=True)
class ExplicitQueryComposer:
    mode: str
    separator: str

    def __post_init__(self) -> None:
        if self.mode not in {"summary_only", "original_and_summary"} or not isinstance(self.separator, str):
            raise ValueError("query composition mode and separator must be explicitly resolved")

    def compose(self, original_query: str, summary: str) -> str:
        if self.mode == "summary_only":
            return summary
        if self.mode == "original_and_summary":
            return original_query + self.separator + summary
        raise ValueError("query composition must be resolved from source")


def minmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Explicit adapter option, NOT established as the author's implementation."""
    if not scores:
        return {}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("nonfinite retrieval score")
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _fusion(dense: Mapping[str, float], sparse: Mapping[str, float], *,
                normalization: Callable[[Mapping[str, float]], Mapping[str, float]],
                excluded_ids: frozenset[str], exclude_before_normalization: bool,
                top_k: int):
    """Diver §2.3 equal-weight fusion. Caller specifies normalization semantics."""
    if type(top_k) is not int or top_k <= 0 or set(dense) != set(sparse):
        raise ValueError("positive top_k and matching complete scoring domains required")
    if type(exclude_before_normalization) is not bool:
        raise ValueError("exclude_before_normalization must be an explicit boolean")
    if any(not isinstance(key, str) or not key for key in chain(dense, excluded_ids)):
        raise ValueError("document identities must be nonempty strings")
    if any(isinstance(value, bool) or not math.isfinite(value)
           for value in chain(dense.values(), sparse.values())):
        raise ValueError("raw retrieval scores must be finite numeric values")
    candidates = set(dense) - excluded_ids
    domain = candidates if exclude_before_normalization else set(dense)
    ordered_domain = sorted(domain)
    left = dict(normalization({k: dense[k] for k in ordered_domain}))
    right = dict(normalization({k: sparse[k] for k in ordered_domain}))
    if set(left) != domain or set(right) != domain:
        raise ValueError("normalization changed the scoring domain")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in chain(left.values(), right.values())):
        raise ValueError("Diver requires normalized scores in [0,1]")
    fused = {k: 0.5 * left[k] + 0.5 * right[k] for k in candidates}
    ids = sorted(fused, key=lambda k: (-fused[k], k))[:top_k]
    ranked = tuple(RankedDocument(k, fused[k], dense_raw=dense[k], sparse_raw=sparse[k],
                                  dense_normalized=left[k], sparse_normalized=right[k]) for k in ids)
    return {"dense_raw": dense, "sparse_raw": sparse, "dense_normalized": left,
            "sparse_normalized": right, "fused": fused, "ranking": ranked}


def hybrid_rank(dense, sparse, **kwargs) -> tuple[RankedDocument, ...]:
    return _fusion(dense, sparse, **kwargs)["ranking"]


class FixedHybridRetriever:
    """Scorers and observation rendering are supplied, versioned dependencies."""

    def __init__(self, *, dense_score: Callable, sparse_score: Callable,
                 corpus: Mapping[str, str], normalization: Callable,
                 exclude_before_normalization: bool, top_k: int,
                 render_observation: Callable[[Sequence[RankedDocument]], Sequence[int]],
                 backend_version: str):
        if (not isinstance(backend_version, str) or not backend_version
                or type(top_k) is not int or top_k <= 0
                or type(exclude_before_normalization) is not bool):
            raise ValueError("retriever version, positive top_k and explicit exclusion mode are required")
        if any(not isinstance(key, str) or not key or not isinstance(value, str)
               for key, value in corpus.items()):
            raise ValueError("corpus requires nonempty string IDs and string document text")
        self.dense_score, self.sparse_score = dense_score, sparse_score
        self.corpus, self.normalization = dict(corpus), normalization
        self.exclude_before_normalization = exclude_before_normalization
        self.top_k, self.render_observation = top_k, render_observation
        self.backend_version = backend_version

    def recipe(self):
        """Bind the scorers actually in use, including normalization and cutoffs."""
        return {
            "dense": self.dense_score.recipe(), "sparse": self.sparse_score.recipe(),
            "candidate_domain": "all_domain_corpus_rows", "backend_version": self.backend_version,
            "normalization": self.normalization.__name__,
            "normalization_source_sha256": source_sha256(self.normalization),
            "fusion_source_sha256": source_sha256(_fusion), "weights": [0.5, 0.5],
            "ties": "score_desc_then_document_id_lexicographic_ascending",
            "exclusion": "fixed_dataset_ids_each_query",
            "exclude_before_normalization": self.exclude_before_normalization,
            "cutoff": self.top_k, "reward": "pytrec_eval.ndcg_cut_10_raw_fraction",
            "packages": package_versions(("numpy", "gensim", "scipy", "pyserini", "pyjnius", "pytrec-eval")),
        }

    def score_trace(self, query: str, *, excluded_ids: frozenset[str]):
        if not isinstance(query, str) or not query:
            raise ValueError("retrieval query must be a nonempty string")
        dense, sparse = self.dense_score(query), self.sparse_score(query)
        if set(dense) != set(self.corpus) or set(sparse) != set(self.corpus):
            raise ValueError("scorer/corpus mismatch; no silent candidate-pool change")
        return _fusion(dense, sparse, normalization=self.normalization, excluded_ids=excluded_ids,
                       exclude_before_normalization=self.exclude_before_normalization, top_k=self.top_k)

    def retrieve(self, query: str, *, excluded_ids: frozenset[str]) -> RetrievalResult:
        ranking = self.score_trace(query, excluded_ids=excluded_ids)["ranking"]
        documents = tuple(replace(d, text=self.corpus[d.document_id]) for d in ranking)
        observation = tuple(self.render_observation(documents))
        if any(type(t) is not int or t < 0 for t in observation):
            raise ValueError("invalid observation token IDs")
        return RetrievalResult(query, documents, observation, self.backend_version)


class PyTrecReward:
    """Official pytrec_eval ndcg_cut_10, raw [0,1], never report-round rewards."""

    def __init__(self, qrels: Mapping[str, Mapping[str, int]]):
        import pytrec_eval
        self.qrels = {q: dict(rels) for q, rels in qrels.items()}
        self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {"ndcg_cut_10"})

    def score(self, query_id: str, ranked_ids: Sequence[str]) -> float:
        if query_id not in self.qrels:
            raise ValueError("unknown qrels query")
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("duplicate ranking document")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ranked_ids):
            raise ValueError("ranked document IDs must be nonempty strings")
        if not ranked_ids:
            return 0.0
        scores = {doc_id: float(len(ranked_ids) - rank) for rank, doc_id in enumerate(ranked_ids)}
        result = self.evaluator.evaluate({query_id: scores})
        reward = float(result[query_id]["ndcg_cut_10"])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("pytrec_eval returned an invalid fractional NDCG@10")
        return reward


def last_token_embeddings(model, input_ids, attention_mask):
    """Original Diver model card: last nonpadding token, L2 normalization."""
    import torch
    import torch.nn.functional as functional
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("expected matching [batch, sequence] inputs")
    if getattr(model, "training", True):
        raise ValueError("fixed retrieval encoder must remain in eval mode")
    if input_ids.device != attention_mask.device or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention mask must be binary and share the input device")
    if not bool(attention_mask.any(dim=1).all()):
        raise ValueError("empty encoder input")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    last = positions.expand_as(input_ids).masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    with torch.inference_mode():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError("encoder hidden states must align with native inputs")
        vectors = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        vectors = vectors.float()
        norms = vectors.norm(dim=-1)
        if (not bool(torch.isfinite(vectors).all()) or not bool(torch.isfinite(norms).all())
                or not bool((norms > 0).all())):
            raise ValueError("encoder returned a nonfinite or zero document vector")
        return functional.normalize(vectors, p=2, dim=-1)


class DenseVectorScorer:
    """Exact cosine scores over a fixed, pre-encoded corpus; no ANN approximation.

    The .npy matrix may be memory mapped. Chunking changes memory use only.
    Corpus encoding is a separate Slurm stage, never an implicit download/build.
    """

    def __init__(self, document_ids, vectors, encode_query: Callable, *, chunk_size: int):
        import numpy as np
        self.ids = tuple(document_ids)
        if (not self.ids or any(not isinstance(identity, str) or not identity for identity in self.ids)
                or len(set(self.ids)) != len(self.ids) or vectors.ndim != 2 or vectors.shape[1] == 0
                or vectors.shape[0] != len(self.ids) or type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError("invalid corpus vectors, identities or chunk size")
        if not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError("corpus vectors must be floating point")
        self.vectors, self.encode_query, self.chunk_size = vectors, encode_query, chunk_size
        # Corpus vectors are an immutable, pre-encoded retrieval asset.  Scan
        # them once at load time using the same float32 blocks used for exact
        # scoring, then retain only their norms.  Query-time validation must
        # still cover the freshly encoded query, but must not repeatedly scan
        # the fixed corpus merely to rediscover these norms.
        self.document_norms = np.empty(len(self.ids), dtype=np.float32)
        for start in range(0, len(self.ids), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.ids))
            block = np.asarray(self.vectors[start:stop], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or not np.isfinite(norms).all() or not (norms > 0).all():
                raise ValueError("invalid document embedding")
            self.document_norms[start:stop] = norms

    def recipe(self):
        return {"method": "exact_float32_cosine", "chunk_rows": self.chunk_size,
                "source_sha256": source_sha256(type(self))}

    def __call__(self, query: str) -> dict[str, float]:
        import numpy as np
        query_vector = np.asarray(self.encode_query(query), dtype=np.float32)
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.shape != (self.vectors.shape[1],) or not np.isfinite(query_vector).all():
            raise ValueError("query embedding has incorrect shape or nonfinite values")
        query_norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_norm) or not query_norm > 0:
            raise ValueError("zero query vector or nonfinite query norm")
        query_vector = query_vector / query_norm
        result = {}
        for start in range(0, len(self.ids), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.ids))
            block = np.asarray(self.vectors[start:stop], dtype=np.float32)
            scores = (block @ query_vector) / self.document_norms[start:stop]
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite cosine score")
            result.update(zip(self.ids[start:start + len(scores)], map(float, scores), strict=True))
        return result


class LuceneBM25Scorer:
    """Diver's Lucene analyzer + Gensim LuceneBM25Model, k1=.9, b=.4.

    This independent adapter follows the pinned retrievers.py implementation,
    including BM25-transforming both corpus and query. Construction tokenizes
    the corpus and builds a sparse matrix: it is compute and requires Slurm.
    Dependencies/JVM must already exist; this class never installs them.
    """

    def __init__(self, corpus: Mapping[str, str]):
        require_slurm()
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        from pyserini import analysis
        if not corpus:
            raise ValueError("empty BM25 corpus")
        self.ids = tuple(corpus)
        java_analyzer = analysis.get_lucene_analyzer()
        self.analyzer_class = str(java_analyzer.getClass().getName())
        self.analyzer = analysis.Analyzer(java_analyzer)
        tokens = [self.analyzer.analyze(corpus[key]) for key in self.ids]
        self.analyzed_sha256 = content_sha256([content_sha256(row) for row in tokens])
        self.dictionary = Dictionary(tokens)
        if len(self.dictionary) == 0:
            raise ValueError("corpus has no analyzed terms")
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        weighted = self.model[[self.dictionary.doc2bow(row) for row in tokens]]
        self.index = SparseMatrixSimilarity(weighted, num_docs=len(tokens),
                                             num_terms=len(self.dictionary),
                                             normalize_queries=False, normalize_documents=False)

    def recipe(self):
        return {"method": "gensim.LuceneBM25Model", "k1": 0.9, "b": 0.4,
                "analyzer": "pyserini.analysis.get_lucene_analyzer(defaults)",
                "analyzer_class": self.analyzer_class, "analyzed_rows_sha256": self.analyzed_sha256,
                "transform_queries": True, "transform_documents": True,
                "index_normalize_queries": False, "index_normalize_documents": False,
                "source_sha256": source_sha256(type(self))}

    def __call__(self, query: str) -> dict[str, float]:
        weighted = self.model[self.dictionary.doc2bow(self.analyzer.analyze(query))]
        values = self.index[weighted].tolist()
        return dict(zip(self.ids, map(float, values), strict=True))


class DiverEncoder:
    """Original Diver model-card encoding, supplied offline model/tokenizer.

    Query instruction is pinned; document instruction and truncation length are
    explicit because upstream artifacts use different document prompt profiles.
    """
    QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

    def __init__(self, model, tokenizer, *, device, max_length: int, document_prefix: str):
        if model.training or type(max_length) is not int or max_length <= 0:
            raise ValueError("Diver must be in eval mode with an explicit length")
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_length, self.document_prefix = max_length, document_prefix
        self._identity = None

    @classmethod
    def from_local(cls, deployment):
        """Only this loader produces a receipt-capable encoder."""
        require_slurm()
        from .environment import configure_caches
        configure_caches(deployment["cache_root"])
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = require_cluster_storage(deployment["retriever_model_path"])
        assets = local_model_identity(path)
        dtype_name = deployment["dtype"]
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
        args = dict(local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(str(path), **args)
        model, load_info = AutoModel.from_pretrained(str(path), torch_dtype=dtype,
                                                     output_loading_info=True, **args)
        if any(load_info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("incomplete local encoder weight load")
        device = deployment["retriever_device"]
        model.to(device).eval()
        model.requires_grad_(False)
        if model.dtype != dtype or local_model_identity(path) != assets:
            raise ValueError("retriever precision mismatch or snapshot changed during load")
        encoder = cls(model, tokenizer, device=device, max_length=deployment["retriever_max_length"],
                      document_prefix=deployment["document_prefix"])
        if not isinstance(encoder.document_prefix, str):
            raise ValueError("explicit document prefix required")
        impl = {
            "encoder_source_sha256": source_sha256(cls, last_token_embeddings),
            "model": implementation_class(model), "tokenizer": implementation_class(tokenizer),
            "packages": package_versions(("torch", "transformers", "tokenizers", "safetensors", "numpy")),
            "dtype": dtype_name, "output_dtype": "float32", "hidden_size": model.config.hidden_size,
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "pooling": "last_nonpadding_token", "normalization": "float32_l2_last_dimension",
            "query_prefix": encoder.QUERY_PREFIX, "document_prefix": encoder.document_prefix,
            "padding": "longest_in_batch", "padding_side": tokenizer.padding_side,
            "truncation": True, "truncation_side": tokenizer.truncation_side,
            "max_length": encoder.max_length, "add_special_tokens": True,
            "all_special_ids": tokenizer.all_special_ids, "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": content_sha256(getattr(tokenizer, "chat_template", None)),
        }
        encoder._identity = {"assets": assets, "implementation": impl,
                             "sha256": content_sha256([assets, impl])}
        return encoder

    @property
    def identity(self):
        if self._identity is None:
            raise ValueError("vector bundles require an encoder loaded by DiverEncoder.from_local")
        return self._identity

    def tokenize(self, texts: Sequence[str], *, is_query: bool):
        prefix = self.QUERY_PREFIX if is_query else self.document_prefix
        return self.tokenizer([prefix + text for text in texts], padding=True,
                              truncation=True, max_length=self.max_length,
                              add_special_tokens=True, return_tensors="pt")

    def encode_inputs(self, inputs):
        vectors = last_token_embeddings(self.model, inputs["input_ids"].to(self.device),
                                        inputs["attention_mask"].to(self.device))
        return vectors.cpu().numpy()

    def encode(self, texts: Sequence[str], *, is_query: bool):
        return self.encode_inputs(self.tokenize(texts, is_query=is_query))

    def query(self, text: str):
        return self.encode([text], is_query=True)[0]

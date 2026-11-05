"""Offline, single-process reference RICE-PO training driver.

This module is source only: imports do not load torch, transformers, models, or
data. ``run`` requires Slurm and pre-existing local assets. It is an independent
implementation with explicit choices, not a validated four-rank implementation.
Actor and frozen reference share one device; the retriever uses a second device.
Memory feasibility and training behavior have not been runtime-validated.
"""

from __future__ import annotations

from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Sequence
import uuid

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm, stage_receipt
from .backends import TransformersEntropyProvider
from .config import require_execution
from .contracts import Generation, QueryComposer
from .data import QueryExample, content_sha256, project_corpus_document, read_jsonl, validate_split_manifests
from .formats import NativeTagParser, PromptRenderer
from .modeling import causal_token_scores, make_forward
from .retrieval import (
    DenseVectorScorer, DiverEncoder, ExplicitQueryComposer, FixedHybridRetriever,
    LuceneBM25Scorer, PyTrecReward, minmax,
)
from .rollout import BranchRequest, RolloutRules, RolloutRunner
from .training import UpdateItem, ppo_update
from .workflow import IterationComponents, run_iteration


class HFNativeGenerator:
    """Sample untouched model probabilities and retain every native output ID.

    A fresh GenerationConfig plus use_model_defaults=False prevents a model's
    bundled generation recipe from silently applying a different sampling law.
    The explicit law is temperature=1, top_p=1, top_k=0, with no penalties,
    constraints, or logit processors. Stop conditions do not reweight logits.
    """

    def __init__(self, *, model: Any, tokenizer: Any, device: Any,
                 policy_version_getter: Any, max_new_tokens: int,
                 max_context_tokens: int, prefilled_think: bool, precision_context: Any):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._policy_version_getter = policy_version_getter
        self.max_new_tokens, self.max_context_tokens = max_new_tokens, max_context_tokens
        self.precision_context = precision_context
        self.parser = NativeTagParser(tokenizer, prefilled_think=prefilled_think, malformed_policy="raise")
        if "use_model_defaults" not in inspect.signature(model.generate).parameters:
            raise RuntimeError("installed Transformers must support generate(use_model_defaults=False)")

    @property
    def policy_version(self) -> str:
        return self._policy_version_getter()

    def generate(self, prefix_token_ids: Sequence[int], *, seed: int) -> Generation:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        prefix = tuple(prefix_token_ids)
        if not prefix or any(type(token) is not int or token < 0 for token in prefix):
            raise ValueError("generation requires nonempty native integer token IDs")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("generation seed must lie in [0, 2**63)")
        if self.model.training:
            raise ValueError("reference runtime requires eval mode, including during PPO")
        remaining = self.max_context_tokens - len(prefix)
        if remaining <= 0:
            raise ValueError("generation cannot truncate or exceed the native context")
        limit = min(self.max_new_tokens, remaining)
        tokenizer, prefix_length = self.tokenizer, len(prefix)

        class SummaryClosed(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Decode only to detect termination; never reconstruct token IDs.
                text = tokenizer.decode(input_ids[0, prefix_length:].tolist(), skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                return torch.tensor(["</summary>" in text], device=input_ids.device, dtype=torch.bool)

        recipe = GenerationConfig(
            do_sample=True, num_beams=1, num_return_sequences=1,
            max_new_tokens=limit, min_length=0, min_new_tokens=0,
            temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0, min_p=None,
            epsilon_cutoff=0.0, eta_cutoff=0.0,
            repetition_penalty=1.0, encoder_repetition_penalty=1.0,
            no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
            bad_words_ids=None, force_words_ids=None, forced_bos_token_id=None,
            forced_eos_token_id=None, forced_decoder_ids=None, suppress_tokens=None,
            begin_suppress_tokens=None, sequence_bias=None,
            exponential_decay_length_penalty=None, renormalize_logits=False,
            remove_invalid_values=False, guidance_scale=None, watermarking_config=None,
            token_healing=False, use_cache=True,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            return_dict_in_generate=False, output_scores=False, output_logits=False,
        )
        inputs = torch.tensor([prefix], dtype=torch.long, device=self.device)
        device = torch.device(self.device)
        cuda_indices = [device.index] if device.type == "cuda" else []
        version = self.policy_version
        with torch.random.fork_rng(devices=cuda_indices), torch.inference_mode(), self.precision_context():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=recipe, use_model_defaults=False,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=StoppingCriteriaList([SummaryClosed()]),
                synced_gpus=False,
            )
        if self.policy_version != version:
            raise RuntimeError("actor policy changed during a frozen-policy generation")
        if output.ndim != 2 or output.shape[0] != 1 or tuple(output[0, :prefix_length].tolist()) != prefix:
            raise ValueError("model.generate altered the native prefix or completion count")
        native = tuple(output[0, prefix_length:].tolist())
        text = tokenizer.decode(native, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reason = "stop" if "</summary>" in text else "incomplete"
        return self.parser(native, reason)


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def _local_file(value: Any, name: str, *, large: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must identify an existing local file")
    path = require_cluster_storage(value) if large else Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _configure_caches(cache_root: Path) -> None:
    locations = {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "transformers",
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    }
    for variable, suffix in locations.items():
        destination = cache_root / suffix
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Python may have cached its platform temp directory before these variables.
    import tempfile
    tempfile.tempdir = str(cache_root / "tmp")
    java_tmp = f'"-Djava.io.tmpdir={cache_root / "tmp"}"'
    os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " " + java_tmp).strip()


def _manifest(path: Path, domains: set[str]) -> tuple[list[dict], dict[str, QueryExample], dict[str, dict[str, int]]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    examples, qrels = {}, {}
    for row in rows:
        domain, query_id, query = row["domain"], row["query_id"], row["query"]
        if (domain not in domains or not isinstance(query_id, str) or not query_id
                or not isinstance(query, str) or not query.strip()):
            raise ValueError("manifest requires known domain, nonempty query_id and query")
        # JSON encoding prevents collisions between domain/query IDs containing punctuation.
        identity = json.dumps([domain, query_id], ensure_ascii=False, separators=(",", ":"))
        excluded = row.get("excluded_ids", [])
        if not isinstance(excluded, list) or any(not isinstance(x, str) or not x for x in excluded):
            raise ValueError("excluded_ids must be a list of dataset document IDs")
        relevance = row.get("qrels")
        if (not isinstance(relevance, dict) or not relevance
                or any(not isinstance(doc, str) or not doc or type(grade) is not int or grade < 0
                       for doc, grade in relevance.items()) or not any(relevance.values())):
            raise ValueError("manifest qrels must contain integer grades and a positive judgment")
        if any(grade > 0 and doc in excluded for doc, grade in relevance.items()):
            raise ValueError("positively judged documents may not also be dataset exclusions")
        if identity in examples:
            raise ValueError("duplicate domain/query identity in manifest")
        examples[identity] = QueryExample(identity, domain, query, frozenset(excluded))
        qrels[identity] = dict(relevance)
    return rows, examples, qrels


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(directory.glob("*.py"))}


def _model_source(path: Path) -> dict:
    """Hash local tokenizer/config/weight assets inside Slurm for later holdout verification."""
    suffixes = {".safetensors", ".bin", ".json", ".model", ".txt", ".jinja", ".tiktoken"}
    files = {str(asset.relative_to(path)): file_sha256(asset) for asset in sorted(path.rglob("*"))
             if asset.is_file() and asset.suffix in suffixes and not any(
                 part.startswith(".") for part in asset.relative_to(path).parts)}
    return {"path": str(path), "config_sha256": file_sha256(path / "config.json"), "asset_sha256": files}



def _domain_retriever(domain, spec, deployment, settings, model_spec, encoder, renderer, all_examples, all_qrels):
    """Build the fixed training/holdout backend from identical corpus/vector artifacts."""
    import numpy as np
    paths = {key: _local_file(spec[key], f"{domain}.{key}")
             for key in ("corpus_jsonl", "corpus_vectors_npy", "vector_document_ids_json")}
    corpus = {}
    for row in read_jsonl(paths["corpus_jsonl"]):
        identity, text = project_corpus_document(row)
        if not isinstance(identity, str) or not identity or identity in corpus or not isinstance(text, str):
            raise ValueError("corpus requires unique nonempty string IDs and string text")
        corpus[identity] = text
    ids = json.loads(paths["vector_document_ids_json"].read_text(encoding="utf-8"))
    corpus_ids = set(corpus)
    if (not isinstance(ids, list) or any(not isinstance(identity, str) for identity in ids)
            or len(set(ids)) != len(ids) or set(ids) != corpus_ids):
        raise ValueError("vector row IDs must exactly cover the corpus without duplicates")
    for query_id, example in all_examples.items():
        if example.domain == domain and not set(all_qrels[query_id]) <= corpus_ids:
            raise ValueError("qrels reference documents missing from the domain corpus")
    vectors = np.load(paths["corpus_vectors_npy"], mmap_mode="r", allow_pickle=False)
    receipt_path = _local_file(str(paths["corpus_vectors_npy"].with_suffix(".receipt.json")),
                               f"{domain}.encoding_receipt")
    encoding_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    required_receipt = {
        "schema_version": 1, "domain": domain, "documents": len(corpus),
        "corpus_sha256": hashes["corpus_jsonl"], "vectors_sha256": hashes["corpus_vectors_npy"],
        "document_ids_sha256": hashes["vector_document_ids_json"],
        "model_id": model_spec["retriever_id"], "declared_model_revision": model_spec["retriever_revision"],
        "document_prefix": deployment["document_prefix"], "max_length": deployment["retriever_max_length"],
        "encoder_dtype": deployment["dtype"], "vector_dtype": "float32",
    }
    for key, expected in required_receipt.items():
        if encoding_receipt.get(key) != expected:
            raise ValueError(f"{domain} encoding receipt mismatch: {key}")
    if vectors.dtype != np.dtype("float32"):
        raise ValueError("encoded document matrix dtype does not match float32 receipt")
    dense = DenseVectorScorer(ids, vectors, encoder.query, chunk_size=deployment["dense_chunk_size"])
    sparse = LuceneBM25Scorer(corpus)
    if not isinstance(spec.get("backend_version"), str) or not spec["backend_version"]:
        raise ValueError("every domain requires an explicit fixed backend_version")
    backend = FixedHybridRetriever(
        dense_score=dense, sparse_score=sparse, corpus=corpus, normalization=minmax,
        exclude_before_normalization=settings["excluded_before_normalization"],
        top_k=deployment["retrieval_top_k"], render_observation=renderer.observation,
        backend_version=spec["backend_version"],
    )
    return backend, {
        "backend_version": spec["backend_version"],
        "source_sha256": hashes, "encoding_receipt_sha256": file_sha256(receipt_path),
        "document_count": len(corpus), "vector_shape": list(vectors.shape),
    }


def run(config: dict, deployment: dict) -> dict:
    """Run an explicitly authorized Slurm training budget using local assets.

    Both CLI and API require explicit execution enablement. The API also
    checks Slurm, local-only assets, fixed sampling, and /file_storage01 storage.
    A batch means 128 *queries*, each with N original trajectories. Each update
    samples 128 distinct training queries; different updates may reuse queries.
    """
    require_execution(config)
    job_id = require_slurm()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("this driver supports one process only; it is not an FSDP launch target")
    settings, constants = config["required_research_settings"], config["paper_constants"]
    if config.get("batch_unit") != "queries":
        raise ValueError("reference runtime requires explicit batch_unit='queries'")
    supported = {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "optimize_counterfactual_branches": False,
        "trajectory_std_convention": "population", "kl_estimator": "exact_categorical",
        "optimizer": "adamw", "hybrid_normalization": "minmax",
        "termination_policy": "full_depth_5", "format_failure_policy": "raise",
        "native_span_parser": "strict_native_tag_boundaries_prefilled_think",
        "document_encoding_profile": "original_model_card_raw_documents",
    }
    for key, required in supported.items():
        if settings.get(key) != required:
            raise ValueError(f"reference runtime supports {key}={required!r}; it never overrides configuration")
    seeds = settings["training_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int or not 0 <= seeds[0] < 2**63:
        raise ValueError("one explicit nonnegative training seed per reference run is required")
    updates = _positive_integer(settings["total_updates"], "total_updates")
    seed = seeds[0]
    if settings["checkpoint_selection"] not in {"last", "best_development_macro"}:
        raise ValueError("unsupported checkpoint selection rule")
    if type(settings["excluded_before_normalization"]) is not bool:
        raise ValueError("excluded_before_normalization must be boolean")
    composition = settings["query_composition"]
    if (not isinstance(composition, dict) or composition.get("mode") not in {"summary_only", "original_and_summary"}
            or not isinstance(composition.get("separator"), str)):
        raise ValueError("query_composition requires explicit supported mode and separator")
    optimizer_options = settings["optimizer_hyperparameters"]
    if not isinstance(optimizer_options, dict) or set(optimizer_options) != {"betas", "eps", "weight_decay"}:
        raise ValueError("AdamW requires explicit betas, eps, and weight_decay, with no hidden LR override")
    betas = optimizer_options["betas"]
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x < 1 for x in betas)):
        raise ValueError("AdamW betas must be two finite values in [0,1)")
    for key, strict in (("eps", True), ("weight_decay", False)):
        value = optimizer_options[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (strict and value == 0):
            raise ValueError(f"invalid AdamW {key}")
    output_root = require_cluster_storage(deployment["output_root"])
    cache_root = require_cluster_storage(deployment["cache_root"])
    if output_root.exists():
        raise FileExistsError("output_root must be new; implicit resume or overwrite is forbidden")
    model_paths = {}
    for key in ("actor_model_path", "retriever_model_path"):
        path = require_cluster_storage(deployment[key])
        if not path.is_dir():
            raise FileNotFoundError(path)
        model_paths[key] = path
    manifest_paths = {}
    for name in ("train", "development"):
        path = _local_file(deployment[f"{name}_manifest_path"], f"{name}_manifest_path")
        configured = settings[f"{name}_split_manifest"]
        if not isinstance(configured, str) or Path(configured).resolve() != path:
            raise ValueError(f"config and deployment must identify the same {name} manifest")
        manifest_paths[name] = path
    prompt_path = _local_file(deployment["prompt_path"], "prompt_path", large=False)
    declared_prompt = Path(settings["prompt_artifact"])
    if not declared_prompt.is_absolute():
        declared_prompt = Path(__file__).resolve().parents[2] / declared_prompt
    if not declared_prompt.is_file() or file_sha256(declared_prompt) != file_sha256(prompt_path):
        raise ValueError("deployment prompt must equal the configured prompt artifact")
    domain_specs = deployment["domains"]
    if not isinstance(domain_specs, dict) or not domain_specs:
        raise ValueError("explicit domain corpus/vector deployments are required")
    expected_domains = deployment["expected_development_domains"]
    if (not isinstance(expected_domains, list) or not expected_domains
            or len(set(expected_domains)) != len(expected_domains) or not set(expected_domains) <= set(domain_specs)):
        raise ValueError("expected_development_domains must explicitly name distinct deployed domains")
    train_rows, train_examples, train_qrels = _manifest(manifest_paths["train"], set(domain_specs))
    dev_rows, dev_examples, dev_qrels = _manifest(manifest_paths["development"], set(domain_specs))
    validate_split_manifests({"train": train_rows, "development": dev_rows})
    if {example.domain for example in dev_examples.values()} != set(expected_domains):
        raise ValueError("development manifest domains do not match the explicit expected domain set")
    batch_queries = constants["training_batch_size"]
    if len(train_examples) < batch_queries:
        raise ValueError("training manifest needs at least 128 distinct queries for the declared query batch")
    for key in ("entropy_chunk_size", "logit_chunk_size", "dense_chunk_size", "retriever_max_length", "retrieval_top_k"):
        _positive_integer(deployment[key], key)
    if deployment["retrieval_top_k"] < max(10, settings["observation_top_k"]):
        raise ValueError("retrieval_top_k must cover NDCG@10 and the observation document count")
    if deployment.get("document_prefix") != "":
        raise ValueError("original_model_card_raw_documents requires an explicit empty document_prefix")
    max_grad_norm = deployment["max_grad_norm"]
    if not isinstance(max_grad_norm, (int, float)) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    _configure_caches(cache_root)

    # All heavy imports and construction follow explicit local/Slurm checks.
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from .evaluation import summarize

    actor_device, retriever_device = torch.device(deployment["actor_device"]), torch.device(deployment["retriever_device"])
    if (actor_device.type != "cuda" or retriever_device.type != "cuda" or actor_device.index is None
            or retriever_device.index is None or actor_device == retriever_device):
        raise ValueError("reference deployment requires two distinct explicit CUDA devices")
    if max(actor_device.index, retriever_device.index) >= torch.cuda.device_count():
        raise RuntimeError("requested reference devices are not visible within the Slurm allocation")
    dtype_name = deployment["dtype"]
    if dtype_name not in {"bfloat16", "float32"}:
        raise ValueError("reference runtime supports bfloat16 or float32; float16 scaling is not implemented")
    dtype = getattr(torch, dtype_name)
    def precision_context():
        # Bound the autocast weight cache to an individual forward/generation.
        # In particular, never reuse old cached weights after optimizer.step().
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dtype_name == "bfloat16"))

    output_root.mkdir(parents=True, exist_ok=False)
    run_identity = uuid.uuid4().hex
    config_hash = atomic_json(output_root / "config.json", config)
    deployment_hash = atomic_json(output_root / "deployment.json", deployment)
    input_hashes = {name: file_sha256(path) for name, path in manifest_paths.items()}
    input_hashes["prompt"] = file_sha256(prompt_path)
    source_hashes = _source_hashes()
    atomic_json(output_root / "sources.json", source_hashes)
    atomic_json(output_root / "train-manifest.json", train_rows)
    atomic_json(output_root / "development-manifest.json", dev_rows)
    atomic_json(output_root / "run-identity.json", {
        "run_identity": run_identity, "slurm_job_id": job_id,
        "config_sha256": config_hash, "deployment_sha256": deployment_hash,
        "input_hashes": input_hashes, "source_sha256": content_sha256(source_hashes),
        "batch_unit": "queries", "queries_per_update": batch_queries,
        "batch_sampling": "uniform_without_replacement_within_update_independent_across_updates",
        "new_branches_per_anchor": settings["total_branches_including_original"] - 1,
        "total_branches_including_original": settings["total_branches_including_original"], "world_size": 1,
        "policy_identity_kind": "run_and_iteration_identity_not_weight_content_hash",
        "validation_status": "reference_runtime_not_previously_executed",
    })
    local_args = dict(local_files_only=True, trust_remote_code=False, cache_dir=str(cache_root / "transformers"))
    actor_tokenizer = AutoTokenizer.from_pretrained(str(model_paths["actor_model_path"]), **local_args)
    retriever_tokenizer = AutoTokenizer.from_pretrained(str(model_paths["retriever_model_path"]), **local_args)
    # AdamW updates at 1e-6 must accumulate in FP32 master parameters. Direct
    # bf16 weight updates would round many small changes away at every step.
    actor = AutoModelForCausalLM.from_pretrained(str(model_paths["actor_model_path"]), torch_dtype=torch.float32, **local_args).to(actor_device).eval()
    reference = AutoModelForCausalLM.from_pretrained(str(model_paths["actor_model_path"]), torch_dtype=torch.float32, **local_args).to(actor_device).eval()
    reference.requires_grad_(False)
    retriever_model = AutoModel.from_pretrained(str(model_paths["retriever_model_path"]), torch_dtype=dtype, **local_args).to(retriever_device).eval()
    retriever_model.requires_grad_(False)
    if actor_tokenizer.eos_token_id is None:
        raise ValueError("actor tokenizer must declare its EOS token")
    renderer = PromptRenderer(
        actor_tokenizer, prompt_path.read_text(encoding="utf-8"),
        observation_top_k=settings["observation_top_k"],
        observation_token_limit=settings["observation_token_limit"], prefill_think=True,
    )
    encoder = DiverEncoder(retriever_model, retriever_tokenizer, device=retriever_device,
                           max_length=deployment["retriever_max_length"], document_prefix="")
    version = {"value": f"{run_identity}/initial"}
    generator = HFNativeGenerator(
        model=actor, tokenizer=actor_tokenizer, device=actor_device,
        policy_version_getter=lambda: version["value"],
        max_new_tokens=settings["max_new_tokens_per_step"],
        max_context_tokens=settings["max_context_tokens"], prefilled_think=True,
        precision_context=precision_context,
    )
    all_examples = {**train_examples, **dev_examples}
    all_qrels = {**train_qrels, **dev_qrels}
    reward = PyTrecReward(all_qrels)
    composer: QueryComposer = ExplicitQueryComposer(composition["mode"], composition["separator"])
    retrievers, runners, domain_receipts = {}, {}, {}
    for domain, spec in domain_specs.items():
        backend, domain_receipts[domain] = _domain_retriever(
            domain, spec, deployment, settings, config["model"], encoder, renderer, all_examples, all_qrels)
        retrievers[domain] = backend
        runners[domain] = RolloutRunner(
            generator=generator, retriever=backend, reward_scorer=reward, query_composer=composer,
            rules=RolloutRules(frozenset({"stop"}), settings["max_new_tokens_per_step"], settings["max_context_tokens"]),
            should_stop=lambda steps: False, final_reward=lambda steps: steps[-1].local_reward,
            excluded_ids_by_query={key: ex.excluded_ids for key, ex in all_examples.items() if ex.domain == domain},
        )
    atomic_json(output_root / "retriever-sources.json", domain_receipts)
    # Model identity here is a local deployment identity; checkpoint weight hashes
    # below provide content verification for every newly written actor checkpoint.
    atomic_json(output_root / "model-sources.json", {
        key: _model_source(path)
        for key, path in model_paths.items()
    })
    optimizer = torch.optim.AdamW(actor.parameters(), lr=constants["actor_learning_rate"],
                                 betas=tuple(betas), eps=optimizer_options["eps"],
                                 weight_decay=optimizer_options["weight_decay"])
    raw_forward = make_forward(reference_model=reference, device=actor_device,
                           logit_chunk_size=deployment["logit_chunk_size"],
                           max_context_tokens=settings["max_context_tokens"],
                           policy_call_kwargs={}, reference_call_kwargs={})
    def forward(model, item):
        with precision_context():
            return raw_forward(model, item)

    raw_entropy = TransformersEntropyProvider(
        model=actor, policy_version_getter=lambda: version["value"], device=actor_device,
        chunk_size=deployment["entropy_chunk_size"], max_context_tokens=settings["max_context_tokens"],
        model_call_kwargs={},
    )
    class EntropyWithPrecision:
        @property
        def policy_version(self):
            return raw_entropy.policy_version

        def token_entropies(self, trajectory):
            with precision_context():
                return raw_entropy.token_entropies(trajectory)

    batch_random, seed_random = random.Random(seed), random.Random(f"generation:{seed}")
    used_seeds: set[int] = set()
    seed_requests: dict[str, tuple[int, ...]] = {}
    active = {"iteration": 0}
    best: dict[str, Any] = {}

    def new_seed() -> int:
        value = seed_random.randrange(2**63)
        while value in used_seeds:
            value = seed_random.randrange(2**63)
        used_seeds.add(value)
        return value

    def request_seeds(identity: str, count: int) -> tuple[int, ...]:
        if identity in seed_requests:
            raise ValueError("duplicate rollout seed request identity")
        values = tuple(new_seed() for _ in range(count))
        seed_requests[identity] = values
        return values

    def initial(example: QueryExample, label: str) -> tuple[int, ...]:
        # D0 is an actual retrieval with the original query; no qrels enter this call.
        result = retrievers[example.domain].retrieve(example.query, excluded_ids=example.excluded_ids)
        atomic_json(output_root / f"iteration-{active['iteration']:06d}" / "initial-evidence" / f"{label}.json",
                    {"query_id": example.query_id, "backend_version": result.backend_version,
                     "ranking": [{"document_id": doc.document_id, "score": doc.score} for doc in result.documents]})
        return renderer.initial(example.query, result.documents)

    def rollout_batch(iteration: int, policy_version: str):
        selected = batch_random.sample(list(train_examples), batch_queries)
        atomic_json(output_root / f"iteration-{iteration:06d}" / "query-batch.json", selected)
        trajectories = []
        for query_number, query_id in enumerate(selected):
            example = train_examples[query_id]
            prefix = initial(example, f"train-q{query_number}")
            for sample in range(settings["initial_trajectories_per_query"]):
                trajectory_id = f"i{iteration}-q{query_number}-n{sample}"
                trajectories.append(runners[example.domain].run(
                    query_id=query_id, trajectory_id=trajectory_id,
                    original_query=example.query, initial_prefix_token_ids=prefix,
                    seeds=request_seeds(trajectory_id, 5), old_policy_version=policy_version,
                ))
        return tuple(trajectories)

    def collect_branches(trajectory, anchor):
        example = train_examples[trajectory.query_id]
        requests = []
        for number in range(1, settings["total_branches_including_original"]):
            identity = f"{trajectory.trajectory_id}-d{anchor.depth}-branch{number}"
            requests.append(BranchRequest(identity, request_seeds(identity, 6 - anchor.depth)))
        return runners[example.domain].branch(
            original_query=example.query, original=trajectory, anchor=anchor,
            requests=tuple(requests),
        )

    def update(trajectories, batch, old_policy_version, iteration_root):
        if version["value"] != old_policy_version:
            raise RuntimeError("old policy changed before its probabilities were cached")
        by_id = {identity: (adv, mask) for identity, adv, mask in zip(
            batch.trajectory_ids, batch.advantages, batch.generated_masks, strict=True)}
        items = []
        # Cache every original sequence before any optimizer mutation; branches
        # never enter this list. The frozen reference remains the initial model.
        with torch.no_grad(), precision_context():
            for trajectory in trajectories:
                log_probs, _ = causal_token_scores(
                    actor, reference, trajectory.token_ids, trajectory.generated_mask,
                    device=actor_device, logit_chunk_size=deployment["logit_chunk_size"],
                    max_context_tokens=settings["max_context_tokens"],
                    policy_call_kwargs={}, reference_call_kwargs={},
                )
                advantages, mask = by_id[trajectory.trajectory_id]
                items.append(UpdateItem(trajectory.trajectory_id, old_policy_version,
                                        trajectory.token_ids, mask, advantages,
                                        tuple(log_probs[0].cpu().tolist())))
        atomic_json(iteration_root / "old-policy-update-items.json", items)
        atomic_json(iteration_root / "training-generation-seeds.json", seed_requests)
        actor.eval()  # eval disables dropout; it does not disable autograd.
        with torch.enable_grad():
            result = ppo_update(
                actor, optimizer, items, policy_version=old_policy_version, forward=forward,
                device=actor_device, clip_epsilon=settings["clip_epsilon"],
                kl_coefficient=constants["actor_kl_coefficient"], loss_reduction=settings["loss_reduction"],
                max_grad_norm=max_grad_norm, distributed=False,
            )
        checkpoint = iteration_root / "checkpoint"
        checkpoint.mkdir(exist_ok=False)
        actor.save_pretrained(checkpoint / "actor", safe_serialization=True)
        actor_tokenizer.save_pretrained(checkpoint / "actor")
        torch.save({"optimizer": optimizer.state_dict(), "iteration": active["iteration"],
                    "seed": seed, "batch_random_state": batch_random.getstate(),
                    "generation_random_state": seed_random.getstate(), "used_generation_seeds": sorted(used_seeds),
                    "torch_cpu_rng_state": torch.random.get_rng_state(),
                    "actor_cuda_rng_state": torch.cuda.get_rng_state(actor_device)}, checkpoint / "optimizer-and-rng.pt")
        checkpoint_hashes = {str(path.relative_to(checkpoint)): file_sha256(path)
                             for path in sorted(checkpoint.rglob("*")) if path.is_file()}
        new_version = f"{run_identity}/update-{active['iteration']:06d}"
        atomic_json(checkpoint / "complete.json", {
            "policy_version": new_version, "files": checkpoint_hashes,
            "config_sha256": config_hash, "source_sha256": content_sha256(source_hashes),
        })
        version["value"] = new_version
        evaluation_rows = []
        development_seeds = {}
        for number, (identity, example) in enumerate(dev_examples.items()):
            # Stable per-query seeds compare checkpoints under the same draws
            # and keep development evaluation from perturbing future training RNG.
            random_source = random.Random(f"development:{seed}:{identity}")
            seeds_for_query = tuple(random_source.randrange(2**63) for _ in range(5))
            development_seeds[identity] = seeds_for_query
            trajectory = runners[example.domain].run(
                query_id=identity, trajectory_id=f"dev-i{active['iteration']}-q{number}",
                original_query=example.query, initial_prefix_token_ids=initial(example, f"dev-q{number}"),
                seeds=seeds_for_query, old_policy_version=new_version,
            )
            raw_id = json.loads(identity)[1]
            evaluation_rows.append({"query_id": raw_id, "domain": example.domain,
                                    "ndcg_at_10": trajectory.final_reward})
            atomic_json(iteration_root / "development-trajectories" / f"{number:06d}.json", trajectory)
        evaluation = summarize(evaluation_rows, expected_domains=expected_domains)
        atomic_json(iteration_root / "development-generation-seeds.json", development_seeds)
        atomic_json(iteration_root / "development-results-by-query.json", evaluation_rows)
        evaluation_digest = atomic_json(iteration_root / "development-evaluation.json", evaluation)
        stage_receipt(iteration_root, "evaluation", policy_version=new_version,
                      artifacts={"development-evaluation.json": evaluation_digest})
        candidate = {"policy_version": new_version, "checkpoint": str(checkpoint),
                     "development_macro": evaluation["macro"], "iteration": active["iteration"]}
        if (not best or settings["checkpoint_selection"] == "last"
                or candidate["development_macro"] > best["development_macro"]):
            best.clear()
            best.update(candidate)
        return {"new_policy_version": new_version, "checkpoint": str(checkpoint),
                "optimizer_result": asdict(result), "development": evaluation,
                "selected_checkpoint_so_far": dict(best)}

    def phase_barrier(previous: str, following: str) -> None:
        torch.cuda.synchronize(actor_device)
        torch.cuda.synchronize(retriever_device)
        atomic_json(output_root / f"iteration-{active['iteration']:06d}" / f"transition-{previous}-{following}.json",
                    {"from": previous, "to": following, "policy_version": version["value"], "world_size": 1})

    components = IterationComponents(rollout_batch, EntropyWithPrecision(), collect_branches, update, phase_barrier)
    for iteration in range(updates):
        active["iteration"] = iteration
        seed_requests.clear()
        run_iteration(
            components, iteration=iteration, policy_version=version["value"], output_root=output_root,
            expected_trajectories_per_query=settings["initial_trajectories_per_query"],
            total_branches=settings["total_branches_including_original"],
            local_normalization=settings["local_advantage_normalization"],
            epsilon=settings["normalization_epsilon"],
        )
    summary = {"completed_updates": updates, "last_policy_version": version["value"],
               "selected_checkpoint": dict(best), "output_root": str(output_root),
               "checkpoint_selection": settings["checkpoint_selection"], "slurm_job_id": job_id}
    atomic_json(output_root / "completed.json", summary)
    return summary


def evaluate_checkpoint(config, deployment, *, checkpoint_dir, test_manifest, output_root):
    """Evaluate the selected completed checkpoint without entering the training loop."""
    from .holdout import evaluate_checkpoint as evaluate
    return evaluate(config, deployment, checkpoint_dir=checkpoint_dir, test_manifest=test_manifest,
                    output_root=output_root)

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
import os
from pathlib import Path
from typing import Any, Sequence

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm
from .backends import TransformersEntropyProvider
from .config import require_execution, resolve
from .audit import attempt, operation, raw_completion, GenerationTruncated, event, fingerprint, stable_seed, cost_report
from .asset_identity import local_model_identity
from .environment import configure_caches as _configure_caches
from .dataset_bundle import verify_bundle, read_frozen_rows
from .preparation import load_vector_bundle
from .policy_identity import identify
from .state import (writer, bind_run, recovery_checkpoint, publish_checkpoint, verify_checkpoint,
                    read_json, complete_stage, selected_so_far, finish_development)
from .contracts import Generation, InvalidGeneration,trajectory_behavior_log_probs
from .data import QueryExample
from .formats import CHAT_TURN, NativeTagParser, PromptRenderer, action_closed, native_summary_mode, turn_protocol, parser_options
from .modeling import sampled_old_log_probs, make_forward
from .retrieval import (
    DenseVectorScorer, DiverEncoder, ExplicitQueryComposer, FixedHybridRetriever,
    LuceneBM25Scorer, PyTrecReward, minmax,
)
from .rollout import BranchRequest, RolloutRules, RolloutRunner
from .training import UpdateItem, ppo_plan_update
from .workflow import IterationComponents, run_iteration


class HFNativeGenerator:
    """Sample untouched model probabilities and retain every native output ID.

    A fresh, explicit GenerationConfig prevents a model's bundled generation
    recipe from silently applying a different sampling law.  Some Transformers
    releases expose ``use_model_defaults`` and others reject it as an unused
    model kwarg; the keyword is therefore used only where it is an explicit
    ``generate`` parameter.  The explicit law is temperature=1, top_p=1,
    top_k=0, with no penalties, constraints, or logit processors. Stop
    conditions do not reweight logits.
    """

    def __init__(self, *, model: Any, tokenizer: Any, device: Any,
                 policy_version_getter: Any, max_new_tokens: int,
                 max_context_tokens: int, prefilled_think: bool, precision_context: Any,
                 summary_mode: str = NativeTagParser.STRICT_TAGGED, structured_stop: bool = False,
                 parser_settings=None, failure_semantics="raise"):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._policy_version_getter = policy_version_getter
        self.max_new_tokens, self.max_context_tokens = max_new_tokens, max_context_tokens
        self.precision_context = precision_context
        self.closed = action_closed if structured_stop else lambda text: "</summary>" in text
        if failure_semantics not in ("raise","invalid_action_terminal_zero_v1"):
            raise ValueError("unsupported failure semantics")
        self.failure_semantics = failure_semantics
        self.generation_protocol='legacy_single_stage_v1'
        if summary_mode=='two_stage_env_summary_v1':
            if not prefilled_think or parser_settings:
                raise ValueError('two-stage protocol has a fixed prefilled, ByteLevel structure; legacy parser settings do not apply')
            from .two_stage import TwoStageFormat
            self.generation_protocol=summary_mode
            self.two_stage_format=TwoStageFormat(tokenizer)
            self.parser=None
        else:
            self.parser = NativeTagParser(tokenizer, prefilled_think=prefilled_think, malformed_policy="raise",
                                          summary_mode=summary_mode, **(parser_settings or {}))
        # A supplied GenerationConfig is authoritative for the explicit fields
        # below.  Transformers 5.5.1 has no literal use_model_defaults
        # parameter and validates arbitrary kwargs as model inputs, so passing
        # the old keyword there aborts before sampling.  Do not infer support
        # from **kwargs: that is precisely the path validated as model kwargs.
        self._supports_use_model_defaults = (
            "use_model_defaults" in inspect.signature(model.generate).parameters
        )

    @property
    def policy_version(self) -> str:
        return self._policy_version_getter()

    def _generation_recipe(self,limit):
        from transformers import GenerationConfig
        tokenizer=self.tokenizer
        return GenerationConfig(
            do_sample=True,num_beams=1,num_return_sequences=1,max_new_tokens=limit,min_length=0,min_new_tokens=0,
            temperature=1.0,top_p=1.0,top_k=0,typical_p=1.0,min_p=None,epsilon_cutoff=0.0,eta_cutoff=0.0,
            repetition_penalty=1.0,encoder_repetition_penalty=1.0,no_repeat_ngram_size=0,encoder_no_repeat_ngram_size=0,
            bad_words_ids=None,force_words_ids=None,forced_bos_token_id=None,forced_eos_token_id=None,
            forced_decoder_ids=None,suppress_tokens=None,begin_suppress_tokens=None,sequence_bias=None,
            exponential_decay_length_penalty=None,renormalize_logits=False,remove_invalid_values=False,
            guidance_scale=None,watermarking_config=None,token_healing=False,use_cache=True,
            bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            return_dict_in_generate=False,output_scores=False,output_logits=False)

    def generate_many(self,prefixes,seeds):
        from .batched_generation import generate_many
        return generate_many(self,prefixes,seeds)

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
        if self.generation_protocol=='two_stage_env_summary_v1':
            from .two_stage import generate_two_stage
            return generate_two_stage(self,prefix,seed)
        remaining = self.max_context_tokens - len(prefix)
        if remaining <= 0:
            raise ValueError("generation cannot truncate or exceed the native context")
        limit = min(self.max_new_tokens, remaining)
        tokenizer, prefix_length = self.tokenizer, len(prefix)
        closed_action = self.closed

        class SummaryClosed(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Decode only to detect termination; never reconstruct token IDs.
                text = tokenizer.decode(input_ids[0, prefix_length:].tolist(), skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                return torch.tensor([closed_action(text)], device=input_ids.device, dtype=torch.bool)

        recipe = self._generation_recipe(limit)
        inputs = torch.tensor([prefix], dtype=torch.long, device=self.device)
        device = torch.device(self.device)
        cuda_indices = [device.index] if device.type == "cuda" else []
        version = self.policy_version
        replay = getattr(self,"completion_replay",None)
        cached = replay(prefix,seed,version) if replay else None
        if cached is not None:
            ids = cached["token_ids"]
            if not ids or len(ids)>limit or any(type(t) is not int or t<0 for t in ids):
                raise ValueError("invalid saved native completion")
            output = torch.tensor([prefix+tuple(ids)],dtype=torch.long,device=self.device)
        else:
            output = self._sample(inputs,recipe,cuda_indices,device,seed,prefix_length,SummaryClosed,
                                  LogitsProcessorList,StoppingCriteriaList)
        native = tuple(output[0, prefix_length:].tolist())
        at_limit = len(native) >= limit
        engine_reason = ("eos" if native and native[-1] == tokenizer.eos_token_id
                         else "length_or_stop" if at_limit else "stopping_criterion")
        if cached is not None and engine_reason != cached["finish_reason"]:
            raise ValueError("saved completion termination differs from the frozen generation configuration")
        raw_completion(native, engine_reason, prefix, seed, version,
                       replay_source=cached["source"] if cached else None)
        if self.policy_version != version:
            raise RuntimeError("actor policy changed during a frozen-policy generation")
        if output.ndim != 2 or output.shape[0] != 1 or tuple(output[0, :prefix_length].tolist()) != prefix:
            raise ValueError("model.generate altered the native prefix or completion count")
        return self._parse_completion(native,engine_reason,at_limit)

    def _sample(self,inputs,recipe,cuda_indices,device,seed,prefix_length,SummaryClosed,
                LogitsProcessorList,StoppingCriteriaList):
        import torch
        with torch.random.fork_rng(devices=cuda_indices), torch.inference_mode(), self.precision_context():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            generate_kwargs = {
                "input_ids": inputs,
                "attention_mask": torch.ones_like(inputs),
                "generation_config": recipe,
                "logits_processor": LogitsProcessorList(),
                "stopping_criteria": StoppingCriteriaList([SummaryClosed()]),
                "synced_gpus": False,
            }
            if self._supports_use_model_defaults:
                generate_kwargs["use_model_defaults"] = False
            return self.model.generate(**generate_kwargs)

    def _parse_completion(self,native,engine_reason,at_limit):
        tokenizer = self.tokenizer
        terminal = len(native)
        while terminal and native[terminal-1] in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            terminal -= 1
        text = tokenizer.decode(native[:terminal], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        closed = self.closed(text)
        # The named functional-smoke adapter accepts an EOS-terminated,
        # unwrapped summary only after the model has closed its think span.
        # It does not accept a length-truncated completion as a valid stop.
        if (getattr(self.parser, "structure_mode", None) != "tagged" and getattr(self.parser, "summary_mode", None) in {NativeTagParser.PREFILLED_UNWRAPPED_SUMMARY,
                                         NativeTagParser.SUMMARY_END_OVERLAP_INCLUSIVE_V1,
                                         NativeTagParser.SUMMARY_END_RIGHT_INCLUSIVE_V1}
                and engine_reason == "eos" and "</think>" in text
                and "<summary>" not in text and "</summary>" not in text):
            closed = True
        reason = "stop" if closed else engine_reason
        event("completion_diagnosis", finish_reason=reason, decoded_preview=text[:400])
        if reason != "stop" and at_limit:
            if self.failure_semantics == "invalid_action_terminal_zero_v1":
                return InvalidGeneration(native,engine_reason,"generation_length",
                                         "generation ceiling reached before a valid action")
            raise GenerationTruncated("generation exhausted its configured length before closing the summary")
        try:
            return self.parser(native, reason)
        except Exception as exc:
            if (self.failure_semantics == "invalid_action_terminal_zero_v1"
                    and getattr(exc,"code",None) in {"tag_structure","empty_span"}):
                return InvalidGeneration(native,engine_reason,exc.code,str(exc))
            raise


def _local_file(value: Any, name: str, *, large: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must identify an existing local file")
    path = require_cluster_storage(value) if large else Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _manifest(path: Path, domains: set[str], *, expected_sha256: str) -> tuple[list[dict], dict[str, QueryExample], dict[str, dict[str, int]]]:
    rows = read_frozen_rows(path, expected_sha256)
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


_model_source = local_model_identity


def _domain_retriever(domain, spec, deployment, settings, model_spec, encoder, renderer, all_examples, all_qrels):
    """One fixed ranking backend for training, development, D0 and interactive test."""
    with operation("asset_binding", domain=domain):
        corpus, ids, vectors, audit = load_vector_bundle(domain, spec, model_spec, encoder)
        for q, example in all_examples.items():
            if example.domain == domain and not set(all_qrels[q]) <= corpus.keys():
                raise ValueError("positive document missing from frozen corpus")
        if deployment["retrieval_top_k"] < max(10, settings.get("observation_top_k",0)):
            raise ValueError("retrieval depth must cover the metric and the model observation")
        dense = DenseVectorScorer(ids, vectors, encoder.query, chunk_size=deployment["dense_chunk_size"],
                                  encode_queries=lambda queries:encoder.encode(queries,is_query=True))
        sparse = LuceneBM25Scorer(corpus)
        backend = FixedHybridRetriever(dense_score=dense, sparse_score=sparse, corpus=corpus,
            normalization=minmax, exclude_before_normalization=settings["excluded_before_normalization"],
            top_k=deployment["retrieval_top_k"], render_observation=renderer.observation if renderer else lambda docs: (),
            backend_version=spec["backend_version"],ranking_impl=deployment.get('retrieval_ranking_impl','reference'),
            query_batch_size=deployment.get('retrieval_query_batch_size',1))
        audit.update(backend_version=spec["backend_version"], ranking=backend.recipe(),
                     query_composition=settings.get("query_composition", {"mode":"original_query_D0"}),
                     observation_top_k=settings.get("observation_top_k",0),
                     observation_token_limit=settings.get("observation_token_limit",0))
    return backend,audit


def initial_evaluation_mode(deployment, *, parent, resume):
    """Validate the fresh, frozen-policy development-only evaluation mode.

    This deliberately has no training or checkpoint semantics: it is used to
    establish the update-0 comparator through the same live rollout backend as
    a candidate run.  A continuation or recovery would make that identity
    ambiguous, so both are refused.
    """
    enabled = deployment.get("initial_evaluation_only", False)
    if type(enabled) is not bool:
        raise ValueError("initial_evaluation_only must be a boolean")
    if enabled and (parent is not None or resume):
        raise ValueError("initial evaluation mode requires a fresh run")
    return enabled


def checkpoint_diagnostic_mode(deployment, plan, *, parent, resume):
    """Explicit fresh one-update timing mode; never publishes development success."""
    enabled=deployment.get('stop_after_checkpoint_diagnostic',False)
    if type(enabled) is not bool:
        raise ValueError('checkpoint diagnostic flag must be a boolean')
    spec=deployment.get('precision_pilot',{})
    if spec.get('candidate')=='SCALE':return spec.get('stage')=='capacity'
    pilot=deployment.get('precision_pilot') is not None
    validation=deployment.get('precision_pilot',{}).get('candidate') in ('G_A','G_B')
    if validation:
        if not enabled or parent is not None or plan['checkpoint_selection']!='last':
            raise ValueError('paired validation requires checkpoint-only mode and immutable same-run recovery')
        return True
    if enabled and (parent is not None or resume or (plan['total_updates']!=1 and not (pilot and plan['total_updates']==2))
                    or plan['checkpoint_selection']!='last'):
        raise ValueError('checkpoint diagnostic requires fresh one-update last-checkpoint selection')
    return enabled


def require_qualified_retrieval_batch(deployment):
    size=deployment.get('retrieval_query_batch_size',1)
    if type(size) is not int or size<1:
        raise ValueError('positive retrieval batch size required')
    if deployment['dtype']=='bfloat16' and size>1:
        raise ValueError('BF16 Diver batch>1 failed exact ranking regression; use qualified batch1')
    return size


def run(config, deployment, *, resume=False):
    require_execution(config, command="train")
    require_slurm()
    from .modeling import _validate_projection
    _validate_projection(deployment.get('hf_projection_mode','full_sequence'))
    if 'hf_matmul_precision' in deployment:
        raise ValueError('HF precision is benchmark-only; HF TF32 failed numerical acceptance. Keep the existing IEEE training path.')
    if int(os.environ.get("WORLD_SIZE","1")) != 1 or deployment.get("runtime") != "single_process_hf_reference":
        raise ValueError("only the one-process reference backend is supported")
    root = require_cluster_storage(deployment["output_root"])
    from contextlib import ExitStack
    with writer(root,resume=resume), attempt(root,recovery=resume) as attempt_dir, ExitStack() as cleanup:
        return _train(config,deployment,root,attempt_dir,resume,cleanup=cleanup)


def _train(config,deployment,root,attempt_dir,resume,*,cleanup):
    from .assessment import evaluate_queries
    resolved = resolve(config,"train")
    plan = resolved["values"]
    from .precision_pilot import pilot_settings
    pilot=pilot_settings(deployment,plan)
    validation=bool(pilot and pilot.get('kind')=='tf32_gemm_training_validation_v1')
    scale=bool(pilot and pilot.get('kind')=='biology_scale_training_v1')
    memory_repair=deployment.get('hf_memory_repair')
    if memory_repair is not None:
        if not scale or pilot['stage']!='training':raise ValueError('memory repair is an explicit scale-training choice only')
        from .scale_memory import validate_memory_repair
        validate_memory_repair(memory_repair,pilot['validation_plan_sha256'])
    checkpoint_diagnostic_mode(deployment,plan,parent=deployment.get('parent_run_root'),resume=resume)
    require_qualified_retrieval_batch(deployment)
    checkpoint_io=deployment.get('checkpoint_io','direct_shared_v1')
    sampling_backend=deployment.get('sampling_backend','serial_hf_v1')
    if sampling_backend not in ('serial_hf_v1','hf_native_wavefront_v1','vllm_ieee_two_stage_v1','vllm_mixed_precision_pilot_v1'):
        raise ValueError('unsupported sampling backend')
    if sampling_backend=='vllm_mixed_precision_pilot_v1' and pilot is None:
        raise ValueError('mixed backend requires an independently screened pilot plan')
    is_vllm=sampling_backend in ('vllm_ieee_two_stage_v1','vllm_mixed_precision_pilot_v1')
    if checkpoint_io not in ('direct_shared_v1','local_verified_v1'):
        raise ValueError('unsupported checkpoint IO mode')
    parser_mode = native_summary_mode(deployment["output_parser_mode"])
    if (parser_mode=='two_stage_env_summary_v1') != (plan['native_span_parser']=='two_stage_env_summary_v1'):
        raise ValueError('two-stage generation must match its explicit research recipe')
    effective_parser = parser_options(deployment)
    if effective_parser.get("boundary_rule") == "native_body_coverage_v1" and plan["native_span_parser"] != "configured_native_body_coverage_v1":
        raise ValueError("resolved recipe differs from deployed native body coverage")
    effective_generation = dict(parser=effective_parser, output_parser_mode=deployment["output_parser_mode"],
        turn_protocol=turn_protocol(deployment), prompt=plan["prompt_artifact"], seed_schedule="controlled_v2",
        failure_semantics=plan["format_failure_policy"])
    if is_vllm:
        from .vllm_two_stage import vllm_runtime_options
        effective_generation['vllm_runtime']=vllm_runtime_options(deployment.get('vllm_runtime'),
            numeric_mode=pilot['numeric_mode'] if pilot else 'ieee_fp32')
        if pilot:effective_generation['precision_pilot']=dict(candidate=pilot['candidate'],
            validation_plan_sha256=pilot['validation_plan_sha256'],historical_failures_preserved=True)
    with operation("asset_binding"):
        if scale:
            from .asset_identity import VerifiedModelIdentity
            asset_identities=VerifiedModelIdentity();actor_identity=asset_identities.get
        else:actor_identity=local_model_identity
        frozen = verify_bundle(deployment)
        domain_specs = deployment["domains"]
        domains = set(domain_specs)
        for name in ("train","development"):
            if Path(plan[f"{name}_split_manifest"]).resolve() != frozen.paths[name]:
                raise ValueError(f"{name} manifest differs from resolved training recipe")
        _, train, train_qrels = _manifest(frozen.paths["train"],domains,expected_sha256=frozen.manifest_sha256["train"])
        _, dev, dev_qrels = _manifest(frozen.paths["development"],domains,expected_sha256=frozen.manifest_sha256["development"])
        for name,rows in (("train",train),("development",dev)):
            if {tuple(json.loads(q)) for q in rows} != frozen.identities[name]:
                raise ValueError(f"{name} identities disagree with the frozen receipt")
        if set(deployment["expected_development_domains"]) != {ex.domain for ex in dev.values()}:
            raise ValueError("development domain list differs from frozen development queries")
        if len(train) < plan["queries_per_update"]:
            raise ValueError("train bundle smaller than configured query batch")
        prompt_path = _local_file(deployment["prompt_path"],"prompt",large=False)
        declared = Path(plan["prompt_artifact"])
        if not declared.is_absolute():
            declared = Path(__file__).resolve().parents[2] / declared
        if file_sha256(declared) != file_sha256(prompt_path):
            raise ValueError("prompt artifact differs from research recipe")
        actor_path = require_cluster_storage(deployment["actor_model_path"])
        cache = require_cluster_storage(deployment["cache_root"])
        _configure_caches(cache)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        actor_device,retriever_device = (torch.device(deployment[key]) for key in ("actor_device","retriever_device"))
        if (actor_device.type != "cuda" or retriever_device.type != "cuda" or actor_device.index is None
                or retriever_device.index is None or actor_device == retriever_device
                or max(actor_device.index,retriever_device.index) >= torch.cuda.device_count()):
            raise ValueError("two distinct visible CUDA devices required")
        if (validation or scale) and (torch.get_float32_matmul_precision()!='highest' or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32 or os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1'):
            raise ValueError('paired validation HF process must remain IEEE FP32')
        precision = deployment.get('actor_precision',deployment['dtype'])
        if sampling_backend=='vllm_ieee_two_stage_v1' and (
                precision!='float32' or parser_mode!='two_stage_env_summary_v1'
                or (plan['temperature'],plan['top_p'],plan['top_k'])!=(1.,1.,0)):
            raise ValueError('vLLM candidate requires frozen IEEE FP32 two-stage T1/p1/k0')
        if pilot and (precision!=('float32' if pilot['candidate'] in ('T','B','G_A','G_B','SCALE') else 'bfloat16')
                      or parser_mode!='two_stage_env_summary_v1' or (plan['temperature'],plan['top_p'],plan['top_k'])!=(1.,1.,0)):
            raise ValueError('pilot actor precision or frozen sampling law differs')
        if precision not in {"float32","bfloat16"}:
            raise ValueError("unsupported precision")
        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16,enabled=precision=="bfloat16")
        local = dict(local_files_only=True,trust_remote_code=False)
        actor_source = actor_identity(actor_path)
        tokenizer = AutoTokenizer.from_pretrained(str(actor_path),**local)
        if tokenizer.eos_token_id is None:
            raise ValueError("missing actor EOS token")
        renderer = PromptRenderer(tokenizer,prompt_path.read_text("utf8"),observation_top_k=plan["observation_top_k"],
                                  observation_token_limit=plan["observation_token_limit"],prefill_think=True)
        encoder = DiverEncoder.from_local(deployment)
        model_sources = {"actor_model_path":actor_source,"retriever_model_path":encoder.identity["assets"]}
        examples, qrels = {**train,**dev},{**train_qrels,**dev_qrels}
        retrievers, sources = {},{}
        for domain,spec in domain_specs.items():
            retrievers[domain],sources[domain] = _domain_retriever(domain,spec,deployment,plan,config["model"],
                                                                 encoder,renderer,examples,qrels)
    seed = plan["training_seeds"][0]
    generation_seed=stable_seed(seed,"biology_scale_generation","capacity" if pilot["stage"]=="capacity" else "training") if scale else seed
    binding = dict(schema_version=2,config_sha256=fingerprint(config),recipe_fingerprint=resolved["fingerprint"],
                   source_sha256=fingerprint(_source_hashes()),deployment_sha256=fingerprint(
                   {k:v for k,v in deployment.items() if k not in {"test_scope","expected_test_domains"}}),
                   dataset_receipt_sha256=frozen.receipt_sha256,manifest_sha256=frozen.manifest_sha256,
                   model_sha256=fingerprint(model_sources),retriever_sha256=fingerprint(sources),prompt_sha256=file_sha256(prompt_path),
                   behavior=dict(root_seed=seed, temperature=plan["temperature"],top_p=plan["top_p"],top_k=plan["top_k"],
                                 repetition_penalty=1.0,model_mode="eval", precision=precision,
                                 initial_actor_payload=actor_source["payload_sha256"],tokenizer_length=len(tokenizer),
                                 special_ids=tokenizer.all_special_ids,chat_template_sha256=fingerprint(tokenizer.chat_template),
                                 prefilled_think=True, output_parser_mode=deployment["output_parser_mode"],
                                 rng="sha256-json-stream-v1"))
    protocol = turn_protocol(deployment)
    if protocol == CHAT_TURN:
        binding["behavior"]["turn_protocol"] = protocol
    binding["behavior"]["parser_settings"] = effective_parser
    binding["behavior"]["prompt_sha256"] = file_sha256(prompt_path)
    binding["behavior"]["seed_schedule"] = "controlled_v2"
    binding["behavior"]["failure_semantics"] = plan["format_failure_policy"]
    if sampling_backend!='serial_hf_v1':
        binding['behavior']['sampling_backend']=sampling_backend
    if pilot:
        binding['behavior']['precision_pilot']=dict(candidate=pilot['candidate'],rollout_numeric=pilot['numeric_mode'],
            teacher_compute=precision,parameter_storage='float32',validation_plan_sha256=pilot['validation_plan_sha256'])
    if scale:binding["behavior"]["scale_sampling_namespace"]="capacity" if pilot["stage"]=="capacity" else "training"
    if scale and pilot['stage']=='training':
        baseline=Path(pilot['common_initial_root'])
        binding['common_initial']=dict(report_sha256=file_sha256(baseline/'initial-development/summary.json'),
            policy_version=read_json(baseline/'initial-evaluation-complete.json')['policy_version'],actor_path=str(actor_path))
    from .continuation import (fresh_schedule, parent_run, scheduled_queries, validate_fresh_schedule,
                               validate_state_position)
    parent = parent_run(deployment,binding,plan,train)
    global_start = parent['lineage']['global_start'] if parent else 0
    if parent:
        binding['continuation'] = parent['lineage']
        frozen_schedule = None
    else:
        frozen_schedule = fresh_schedule(train, seed=seed, updates=plan['total_updates'],
                                         count=plan['queries_per_update'],
                                         manifest_sha256=frozen.manifest_sha256['train'])
        if pilot:
            if frozen.manifest_sha256['train']!=pilot['train_manifest_sha256'] or any(q not in train for b in pilot['query_batches'] for q in b):
                raise ValueError('pilot training schedule/manifest differs')
            frozen_schedule['batches']=pilot['query_batches']
            binding['pilot_frozen_schedule_sha256']=fingerprint(frozen_schedule)
        schedule_path = deployment.get('frozen_schedule_path')
        if schedule_path is not None:
            require_cluster_storage(schedule_path)
            frozen_schedule = validate_fresh_schedule(schedule_path, train, seed=seed,
                                                       updates=plan['total_updates'],
                                                       count=plan['queries_per_update'],
                                                       manifest_sha256=frozen.manifest_sha256['train'])
            binding['fresh_schedule_sha256'] = file_sha256(schedule_path)
    action_recovery=None
    if deployment.get('native_action_replay_sources'):
        if not parent or sampling_backend!='serial_hf_v1' or parser_mode!='two_stage_env_summary_v1':
            raise ValueError('native action recovery supports the serial two-stage continuation only')
        from .action_recovery import VerifiedActionRecovery
        action_recovery=VerifiedActionRecovery(deployment['native_action_replay_sources'],binding,plan)
        binding['native_action_recovery']=action_recovery.provenance
    completed_credit_recovery=None
    if deployment.get('completed_credit_recovery'):
        if not memory_repair or pilot['stage']!='training':raise ValueError('credit recovery requires qualified memory repair')
        from .completed_credit_recovery import CompletedCreditRecovery
        completed_credit_recovery=CompletedCreditRecovery(deployment['completed_credit_recovery'],binding,plan,pilot['query_batches'][0])
        binding['completed_credit_recovery']=completed_credit_recovery.provenance
    bound = bind_run(root,binding,resume=resume)
    fixed_files = {"config.json":config,"deployment.json":deployment,"recipe.json":resolved,
                   "effective-generation.json":effective_generation,
                   "model-sources.json":model_sources,"retriever-sources.json":sources,"sources.json":_source_hashes()}
    for name,value in fixed_files.items():
        if resume:
            if read_json(root/name) != value:
                raise ValueError(f"changed run metadata: {name}")
        else:
            atomic_json(root/name,value)
    restored = None
    if resume:
        if scale and not list(root.glob('iteration-[0-9]*')):
            from .scale_training import verify_initial_state
            cp=root/'initial-state';restored=verify_initial_state(root,bound)
        else:cp, restored = recovery_checkpoint(root,bound)
        if (root/"completed.json").exists():
            finished = read_json(root/"completed.json")
            if not (scale and finished["selected_checkpoint"].get("kind")=="frozen_initial_source"):
                verify_checkpoint(require_cluster_storage(finished["selected_checkpoint"]["checkpoint"]),bound)
            if (finished["selected_checkpoint"] != selected_so_far(root,plan["checkpoint_selection"])
                    or restored["iteration"]+1 != plan["total_updates"]):
                raise ValueError("completed run selection or update position differs")
            return finished
        actor_load = actor_path if restored['iteration']==-1 else cp/"actor"
    else:
        actor_load = parent['checkpoint']/'actor' if parent else actor_path
    def load_actor(path):
        model, info = AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=torch.float32,
                                                           output_loading_info=True,**local)
        if any(info.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")):
            raise ValueError("partial actor weight load")
        return model.to(actor_device).eval()
    actor = load_actor(actor_load)
    reference = load_actor(actor_path)
    reference.requires_grad_(False)
    activation_mode=deployment.get('activation_checkpointing','none')
    activation_statistics=None
    if activation_mode=='eval_layers_v1':
        from .modeling import enable_eval_layer_checkpointing
        activation_statistics=enable_eval_layer_checkpointing(actor)
    elif activation_mode!='none':raise ValueError('unsupported activation checkpointing mode')
    if actor_identity(actor_path) != actor_source:
        raise ValueError("actor/tokenizer assets changed while loading")
    opt = plan["optimizer_hyperparameters"]
    optimizer = torch.optim.AdamW(actor.parameters(),lr=plan["actor_learning_rate"],betas=tuple(opt["betas"]),
                                  eps=opt["eps"],weight_decay=opt["weight_decay"])
    loading = restored or (parent['saved'] if parent else None)
    if loading:
        loading_cp = cp if restored else parent['checkpoint']
        loading_binding = bound if restored else parent['binding']
        state = torch.load(loading_cp/"optimizer-and-rng.pt",map_location="cpu",weights_only=True)
        validate_state_position(state,binding=loading_binding,iteration=loading['iteration'],seed=seed,
                                policy_version=loading['policy_version'])
        if restored and state.get('global_update_index',restored['iteration']) != global_start+restored['iteration']:
            raise ValueError('restored global update position differs')
        optimizer.load_state_dict(state["optimizer"])
        torch.random.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"],actor_device)
        if scale:
            if state.get("query_schedule_position")!=loading["iteration"]+1 or state.get("query_schedule_sha256")!=fingerprint(frozen_schedule):
                raise ValueError("checkpoint query schedule position/identity differs")
            torch.cuda.set_rng_state_all(state["cuda_rng_all"])
        from .checkpoint_io import rng_probe, state_digest
        atomic_json(attempt_dir/'loaded-state.json',dict(binding=loading_binding,
            policy_version=loading['policy_version'],local_iteration=loading['iteration'],
            global_completed_updates=(global_start+loading['iteration']+1) if restored else global_start,
            optimizer_sha256=state_digest(state['optimizer']),rng_probe=rng_probe(state,actor_device),
            optimizer_steps=sorted({int(v['step'].item()) for v in state['optimizer']['state'].values() if 'step' in v})))
        del state
    behavior = binding["behavior"]
    live_validation={}
    def register_policy():
        key,evidence = identify(actor,behavior)
        path = root/"behavior-identities"/(key+".json")
        if path.exists():
            if read_json(path) != evidence:
                raise ValueError("behavior identity collision")
        else:
            atomic_json(path,evidence)
        if scale:
            from .asset_identity import tensor_versions
            live_validation.update(policy=key,evidence=evidence,tensor_versions=tensor_versions(actor))
        return key
    version = {"value":register_policy()}
    if memory_repair:
        allocator=torch.cuda.memory._snapshot().get('allocator_settings',{})
        if allocator.get('expandable_segments') is not True:raise ValueError('HF expandable allocator did not actually activate')
        target=root/('memory-residency-runtime-'+attempt_dir.name+'.json')
        atomic_json(target,dict(mode=memory_repair['mode'],allocator_settings=allocator,
            hf_precision='IEEE FP32',projection_mode='full_sequence',optimizer_moments='CPU until before_clip',
            qualification_sha256=memory_repair['qualification_result_sha256']))
    if loading and version["value"] != loading["policy_version"]:
        raise ValueError("restored live policy differs from committed behavior identity")
    if scale and not resume and pilot['stage']!='initial':
        if completed_credit_recovery:completed_credit_recovery.restore_initial(optimizer,version['value'])
        from .scale_training import publish_initial_state
        publish_initial_state(root,dict(optimizer=optimizer.state_dict(),iteration=-1,seed=seed,binding=bound,
            policy_version=version['value'],cpu_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(actor_device),
            cuda_rng_all=torch.cuda.get_rng_state_all(),global_update_index=-1,query_schedule_position=0,
            query_schedule_sha256=fingerprint(frozen_schedule)))
    generator = HFNativeGenerator(model=actor,tokenizer=tokenizer,device=actor_device,
        policy_version_getter=lambda:version["value"],max_new_tokens=plan["max_new_tokens_per_step"],
        max_context_tokens=plan["max_context_tokens"],prefilled_think=True,precision_context=autocast,
        summary_mode=parser_mode, structured_stop=protocol == CHAT_TURN, parser_settings=effective_parser,
        failure_semantics=plan["format_failure_policy"])
    if action_recovery:generator.action_replay=action_recovery
    if deployment.get("native_replay_sources"):
        from .native_replay import SavedNativeCompletions
        generator.completion_replay = SavedNativeCompletions(deployment["native_replay_sources"],binding)
    if is_vllm:
        if deployment.get('native_replay_sources') or action_recovery:
            raise ValueError('HF replay cannot be relabeled as vLLM sampling')
        from .vllm_lifecycle import PhaseSeparatedGenerator
        generator=PhaseSeparatedGenerator(actor=actor,reference=reference,optimizer=optimizer,tokenizer=tokenizer,
            version=lambda:version['value'],behavior=behavior,device=actor_device,root=attempt_dir/'vllm',
            rpc_root=require_cluster_storage(deployment['vllm_rpc_root']),
            max_new_tokens=plan['max_new_tokens_per_step'],max_context_tokens=plan['max_context_tokens'],
            vllm_runtime=effective_generation['vllm_runtime'],numeric_mode=pilot['numeric_mode'] if pilot else 'ieee_fp32',
            reuse_verified_assets=scale,validated_policy_getter=(lambda:live_validation) if scale else None,
            response_timeout_seconds=43200 if scale else 1800,
            defer_optimizer_residency=bool(memory_repair),isolate_training_allocator=bool(memory_repair))
        cleanup.callback(generator.close)
    reward = PyTrecReward(qrels)
    runners = {domain:RolloutRunner(generator=generator,retriever=backend,reward_scorer=reward,
        query_composer=ExplicitQueryComposer(**plan["query_composition"]),rules=RolloutRules(
        frozenset({"stop"}),plan["max_new_tokens_per_step"],plan["max_context_tokens"],plan["rollout_depth"]),
        should_stop=lambda steps:False,final_reward=lambda steps:steps[-1].local_reward,
        render_next_turn=renderer.next_user_turn if protocol == CHAT_TURN else None,
        failure_semantics=plan["format_failure_policy"],
        excluded_ids_by_query={q:ex.excluded_ids for q,ex in examples.items() if ex.domain==domain})
        for domain,backend in retrievers.items()}
    if initial_evaluation_mode(deployment, parent=parent, resume=resume):
        output = root / "initial-development"
        with operation("initial_development_evaluation",evaluation_scope=True):
            summary = evaluate_queries(
                examples=dev, expected_ids=frozen.identities["development"], split="development",
                phase="initial_development", method=plan["method"], root_seed=seed,
                policy_version=version["value"], rollout_depth=plan["rollout_depth"],
                retrievers=retrievers, runners=runners, renderer=renderer, reward=reward, output=output,
                scope="complete", manifest_sha256=frozen.manifest_sha256["development"],
                backend_fingerprints={d:fingerprint(sources[d]) for d in retrievers},
                samples_per_query=plan["eval_samples_per_query"])
        result = {"status":"initial_development_complete", "policy_version":version["value"],
                  "samples_per_query":plan["eval_samples_per_query"], "summary":summary}
        atomic_json(root / "initial-evaluation-complete.json", result)
        return result
    if scale and pilot["stage"]=="training":
        from .scale_training import bind_initial
        baseline=Path(pilot["common_initial_root"])
        initial_policy=read_json(baseline/"initial-evaluation-complete.json")["policy_version"] if resume else version["value"]
        bind_initial(root,baseline,binding,initial_policy,actor_path)
    scoring = make_forward(reference_model=reference,device=actor_device,logit_chunk_size=deployment["logit_chunk_size"],
                           max_context_tokens=plan["max_context_tokens"],policy_call_kwargs={},reference_call_kwargs={},
                           projection_mode=deployment.get("hf_projection_mode","full_sequence"),
                           saved_tensor_policy='cpu_exact_v1' if memory_repair else 'device')
    def forward(model,item):
        length = max(i for i,masked in enumerate(item.generated_mask) if masked)
        with operation("update_forward",input_tokens=length,model_passes=2,scored_tokens=sum(item.generated_mask)), autocast():
            return scoring(model,item)
    entropy = None
    if plan["method"]=="ricepo":
        provider = TransformersEntropyProvider(model=actor,policy_version_getter=lambda:version["value"],
            device=actor_device,chunk_size=deployment["entropy_chunk_size"],max_context_tokens=plan["max_context_tokens"],
            model_call_kwargs={})
        class ConditionalEntropy:
            @property
            def policy_version(self):
                return provider.policy_version
            def token_entropies(self,t):
                with autocast():
                    return provider.token_entropies(t)
        entropy = ConditionalEntropy()

    def rollout_batch(i,policy_version):
        global_i = global_start+i
        if parent:
            selected = scheduled_queries(train,seed,global_i,plan["queries_per_update"])
            if selected != parent['schedule'][i]:
                raise ValueError('sampling departed from frozen continuation schedule')
        else:
            selected = frozen_schedule['batches'][i]
        directory = root/f"iteration-{i:06d}"
        atomic_json(directory/"query-batch.json",selected)
        pending_by_domain={}
        for q in selected:
            ex = train[q]
            with operation("retrieval_D0", query_id=q,iteration=i):
                initial = retrievers[ex.domain].retrieve(ex.query,excluded_ids=ex.excluded_ids)
                atomic_json(directory/"initial-evidence"/(fingerprint(q)+".json"),
                            {"query":q,"ranking":[asdict(d) for d in initial.documents]})
            prefix = renderer.initial(ex.query,initial.documents)
            for n in range(plan["trajectories_per_query"]):
                identity = f"{global_i}-{fingerprint(q)}-{n}"
                arguments=dict(query_id=q,trajectory_id=identity,original_query=ex.query,
                    initial_prefix_token_ids=prefix,old_policy_version=policy_version,
                    initial_ndcg=reward.score(q,tuple(d.document_id for d in initial.documents)),
                    seeds=tuple(stable_seed(generation_seed,"original",global_i,ex.domain,q,n,d) for d in range(1,plan["rollout_depth"]+1)))
                if sampling_backend=='serial_hf_v1':
                    yield runners[ex.domain].run(**arguments)
                else:
                    from .batched_rollout import OriginalRequest
                    pending_by_domain.setdefault(ex.domain,[]).append(OriginalRequest(**arguments))
        # A wave contains every currently ready original trajectory in one
        # domain.  Retrieval feedback is still consumed before that individual
        # trajectory enters its next wave; grouping does not pre-fill future
        # observations or share any state between requests.
        if pending_by_domain:
            from .batched_rollout import run_many
            for domain,pending in pending_by_domain.items():
                with operation('cross_query_original_wave',iteration=i,domain=domain,
                               request_count=len(pending),configured_max_num_seqs=
                               effective_generation.get('vllm_runtime',{}).get('max_num_seqs',len(pending))):
                    yield from run_many(runners[domain],pending)

    sampled_branches=[]
    def branch_job(t,a):
        if action_recovery:action_recovery.check_anchor(a)
        ex = train[t.query_id]
        requests = [BranchRequest(f"{t.trajectory_id}-d{a.depth}-b{b}",
                    tuple(stable_seed(generation_seed,"branch_v2",ex.domain,t.query_id,t.trajectory_id,a.depth,b,d)
                          for d in range(a.depth,plan["rollout_depth"]+1)))
                    for b in range(1,plan["total_branches_including_original"])]
        return ex.domain,dict(original_query=ex.query,original=t,anchor=a,requests=requests)

    def collect_branches(t,a):
        domain,job=branch_job(t,a)
        result=runners[domain].branch(**job,
                                        batched=sampling_backend!='serial_hf_v1')
        if is_vllm:sampled_branches.extend(result.trajectories)
        return result

    def collect_branches_batch(items):
        """Submit all ready counterfactuals for each retriever as one wave."""
        scheduled={}
        result=[None]*len(items)
        for index,(trajectory,anchor) in enumerate(items):
            domain,job=branch_job(trajectory,anchor)
            scheduled.setdefault(domain,[]).append((index,job))
        for domain,jobs in scheduled.items():
            with operation('cross_query_branch_wave',domain=domain,anchor_count=len(jobs),
                           request_count=sum(len(job['requests']) for _,job in jobs),
                           configured_max_num_seqs=effective_generation['vllm_runtime']['max_num_seqs']):
                collections=runners[domain].branch_many(tuple(job for _,job in jobs))
            for (index,_),collection in zip(jobs,collections,strict=True):
                result[index]=collection
                sampled_branches.extend(collection.trajectories)
        if any(value is None for value in result):
            raise ValueError('cross-anchor scheduler omitted a branch collection')
        return tuple(result)

    def update(originals,batch,old_version,directory):
        if version["value"] != old_version:
            raise ValueError("behavior policy changed before old log probabilities were cached")
        original_mode=deployment.get("original_probability_mode","separate")
        if original_mode not in ("separate","fused_actor_v1"):
            raise ValueError("unknown original probability mode")
        fused=original_mode=="fused_actor_v1"
        if fused and (not is_vllm or pilot is None):
            raise ValueError("fused original checking requires explicit sampled-q risk pilot")
        probability_guard=None
        items = []
        pilot_q=[];pilot_p=[];pilot_adv=[]
        with operation("old_log_probs"), torch.no_grad(), autocast():
            for t,adv,mask in zip(originals,batch.advantages,batch.generated_masks,strict=True):
                if fused:
                    sampled=trajectory_behavior_log_probs(t)
                    if sampled is None:raise ValueError("fused PPO requires actual behavior q")
                    items.append(UpdateItem(t.trajectory_id,old_version,t.token_ids,mask,adv,sampled))
                    continue
                length = max(i for i,flag in enumerate(mask) if flag)
                with operation("old_probability_forward",input_tokens=length,model_passes=1,scored_tokens=sum(mask)):
                    logps = sampled_old_log_probs(actor,t.token_ids,mask,device=actor_device,
                        logit_chunk_size=deployment["logit_chunk_size"],max_context_tokens=plan["max_context_tokens"],
                        policy_call_kwargs={},projection_mode=deployment.get("hf_projection_mode","full_sequence"))
                teacher=tuple(logps[0].cpu().tolist())
                sampled=trajectory_behavior_log_probs(t)
                if sampled is not None:
                    from .probability_check import compare_log_probs
                    check=compare_log_probs(sampled,teacher,mask)
                    atomic_json(directory/'behavior-probability-checks'/(fingerprint(t.trajectory_id)+'.json'),check)
                    if not check['within_bf16_tolerance'] and pilot is None:
                        raise ValueError('sampled/teacher probability mismatch exceeds predeclared tolerance')
                    if pilot:
                        pilot_q.extend(v for v,m in zip(sampled,mask,strict=True) if m)
                        pilot_p.extend(v for v,m in zip(teacher,mask,strict=True) if m)
                        pilot_adv.extend(v for v,m in zip(adv,mask,strict=True) if m)
                items.append(UpdateItem(t.trajectory_id,old_version,t.token_ids,mask,adv,sampled if sampled is not None else teacher))
        atomic_json(directory/"old-policy-update-items.json",items)
        if pilot:
            from .precision_pilot import mismatch_statistics,risk_screen
            original_pilot=None if fused else mismatch_statistics(pilot_q,pilot_p,pilot_adv)
            index=global_start+int(directory.name.split('-')[-1])
            from .training_validation import update_risk_limit
            if scale:
                from .scale_training import risk_limit
                outside_max=risk_limit(index)
                if not fused:atomic_json(directory/"scale-original-probability-risk.json",dict(report=original_pilot,outside_limit=outside_max,pass_risk_screen=risk_screen(original_pilot,outside_max=outside_max)))
                if not fused and not risk_screen(original_pilot,outside_max=outside_max):
                    raise ValueError("scale original probability risk failed; no branch dilution and no optimizer step")
            else:outside_max=update_risk_limit(index) if validation else (.01 if index==0 else .02)
            with torch.no_grad(),autocast():
                def pilot_branch_score(t):
                    scored=sampled_old_log_probs(actor,t.token_ids,t.generated_mask,device=actor_device,
                        logit_chunk_size=deployment['logit_chunk_size'],max_context_tokens=plan['max_context_tokens'],policy_call_kwargs={},projection_mode=deployment.get("hf_projection_mode","full_sequence"))[0].cpu().tolist()
                    q=trajectory_behavior_log_probs(t)
                    qr=[v for v,m in zip(q,t.generated_mask,strict=True) if m]
                    pr=[v for v,m in zip(scored,t.generated_mask,strict=True) if m]
                    check=mismatch_statistics(qr,pr)
                    pilot_q.extend(qr);pilot_p.extend(pr)
                    return check
                if validation or scale:
                    from .branch_audit import select_branch_checks,branch_check_plan,execute_branch_checks
                    options=dict(update_index=index,policy=deployment['branch_probability_audit'])
                    selected=select_branch_checks(sampled_branches,**options)
                    atomic_json(directory/'branch-probability-plan.json',branch_check_plan(sampled_branches,**options))
                    def save_check(t,check,expanded):
                        folder='pilot-expanded-probability-checks' if expanded else 'pilot-branch-probability-checks'
                        atomic_json(directory/folder/(fingerprint(t.trajectory_id)+'.json'),check)
                    checks=execute_branch_checks(sampled_branches,selected,score=pilot_branch_score,
                        save_check=save_check,save_expansion=lambda stage,record:atomic_json(directory/('pilot-expansion-'+stage+'.json'),record),
                        passed=lambda c:risk_screen(c,outside_max=outside_max))
                    checked_branch_count=len(checks)
                else:
                    for t in sampled_branches:
                        atomic_json(directory/'pilot-branch-probability-checks'/(fingerprint(t.trajectory_id)+'.json'),pilot_branch_score(t))
                    checked_branch_count=len(sampled_branches)
            if fused:
                from .fused_probability import FusedOriginalProbabilityGuard
                audited_branch_count=len(sampled_branches)
                def save_original(report):
                    atomic_json(directory/'fused-original-probability-risk.json',dict(report=report,
                        outside_limit=outside_max,pass_risk_screen=risk_screen(report,outside_max=outside_max)))
                def branch_and_identity_gate():
                    if version['value']!=old_version:raise ValueError('policy changed during fused risk accumulation')
                    combined=mismatch_statistics(probability_guard.q+pilot_q,probability_guard.p+pilot_p)
                    passed=risk_screen(combined,outside_max=outside_max)
                    atomic_json(directory/'pilot-probability-risk.json',dict(originals=probability_guard.report,
                        all_actual_tokens=combined,branch_count=audited_branch_count,checked_branch_count=checked_branch_count,
                        outside_limit=outside_max,risk_screen_pass=passed,legacy_thresholds_not_rewritten=True,
                        coverage='complete originals from current Actor plus unchanged branch audit; before optimizer.step'))
                    if not passed:raise ValueError('combined probability risk failed; no optimizer step')
                probability_guard=FusedOriginalProbabilityGuard(outside_max=outside_max,
                    branch_gate=branch_and_identity_gate,save_report=save_original)
            else:
                combined=mismatch_statistics(pilot_q,pilot_p)
                passed=risk_screen(combined,outside_max=outside_max)
                if validation or scale:passed=passed and risk_screen(original_pilot,outside_max=outside_max)
                atomic_json(directory/'pilot-probability-risk.json',dict(originals=original_pilot,all_actual_tokens=combined,
                    branch_count=len(sampled_branches),checked_branch_count=checked_branch_count,outside_limit=outside_max,
                    coverage='originals plus selected branch records; reused prefix positions included',
                    strict_one_percent_pass=risk_screen(combined),risk_screen_pass=passed,legacy_thresholds_not_rewritten=True))
                if not passed:raise ValueError('pilot online probability risk screen failed before optimizer step')
            sampled_branches.clear()
        elif is_vllm:
            from .probability_check import compare_log_probs
            from .branch_audit import branch_check_plan,execute_branch_checks,select_branch_checks
            audit_options=dict(update_index=global_start+int(directory.name.split('-')[-1]),
                policy=deployment.get('branch_probability_audit','all'))
            checked_branches=select_branch_checks(sampled_branches,**audit_options)
            audit_plan=branch_check_plan(sampled_branches,**audit_options)
            atomic_json(directory/'branch-probability-plan.json',audit_plan)
            with torch.no_grad(),autocast():
                def score_branch(t):
                    scored=sampled_old_log_probs(actor,t.token_ids,t.generated_mask,device=actor_device,
                        logit_chunk_size=deployment['logit_chunk_size'],max_context_tokens=plan['max_context_tokens'],policy_call_kwargs={},projection_mode=deployment.get("hf_projection_mode","full_sequence"))
                    return compare_log_probs(trajectory_behavior_log_probs(t),scored[0].cpu().tolist(),t.generated_mask)
                def save_branch_check(t,check,expanded):
                    folder='branch-probability-expanded-checks' if expanded else 'branch-probability-checks'
                    atomic_json(directory/folder/(fingerprint(t.trajectory_id)+'.json'),check)
                def save_expansion(stage,record):
                    atomic_json(directory/('branch-probability-expansion-'+stage+'.json'),record)
                execute_branch_checks(sampled_branches,checked_branches,score=score_branch,
                    save_check=save_branch_check,save_expansion=save_expansion)
            sampled_branches.clear()
        with torch.enable_grad():
            observer=comparison=None
            pilot_gradient=None
            if pilot and not scale:
                from .pilot_gradient import prepare
                def ieee_forward(model,item):
                    with torch.autocast('cuda',enabled=False):return scoring(model,item)
                observer,pilot_gradient=prepare(actor,optimizer,items,candidate_forward=forward,
                    ieee_forward=ieee_forward,device=actor_device,recipe=plan,pilot=pilot)
            elif deployment.get('compare_grpo_credit_gradients'):
                if plan['method']!='ricepo':raise ValueError('RICE/GRPO gradient comparison belongs to RICE update only')
                from .credit_gradient import prepare_comparison
                observer,comparison=prepare_comparison(actor,optimizer,originals,items,
                    forward=forward,device=actor_device,recipe=plan)
            if memory_repair:
                previous_observer=observer
                def residency_observer(stage):
                    if stage=='before_clip':generator.prepare_optimizer_step()
                    if previous_observer:previous_observer(stage)
                observer=residency_observer
            diagnostics = ppo_plan_update(actor,optimizer,items,policy_version=old_version,forward=forward,
                                          device=actor_device,recipe=plan,step_observer=observer,probability_guard=probability_guard)
            if fused:sampled_branches.clear()
            if validation or scale:
                diagnostics['hf_precision']=dict(matmul_precision=torch.get_float32_matmul_precision(),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
                if diagnostics['hf_precision']!={'matmul_precision':'highest','matmul_allow_tf32':False,'cudnn_allow_tf32':False}:
                    raise ValueError('HF precision changed during the paired validation')
            if pilot_gradient is not None:
                pilot_gradient['reference_has_gradients']=any(p.grad is not None for p in reference.parameters())
                atomic_json(directory/'pilot-gradient-comparison.json',pilot_gradient)
                diagnostics['pilot_gradient']=pilot_gradient
            if comparison is not None:
                comparison['reference_has_gradients']=any(p.grad is not None for p in reference.parameters())
                atomic_json(directory/'credit-vs-grpo-gradient.json',comparison)
                diagnostics['credit_gradient_comparison']=comparison
        version["value"] = register_policy()
        diagnostics["actual_state_dict_changed"] = version["value"] != old_version
        if scale:optimizer.zero_grad(set_to_none=True)  # committed gradients are no longer needed or offloaded
        if activation_statistics is not None:
            diagnostics['activation_checkpointing']={**activation_statistics,
                'actual_recomputation_observed':activation_statistics['layer_executions']>activation_statistics['checkpoint_forwards'],
                'model_training_mode':actor.training}
        return {"new_policy_version":version["value"],"optimizer":diagnostics}

    def barrier(before,after):
        if is_vllm and after in ('entropy','update'):
            generator.prepare_scoring()
        torch.cuda.synchronize(actor_device)
        torch.cuda.synchronize(retriever_device)

    components = IterationComponents(rollout_batch,entropy,collect_branches if entropy else None,update,barrier,
        collect_branches_batch=collect_branches_batch if is_vllm and entropy else None)
    def finish(i, *, recovered=False):
        directory = root/f"iteration-{i:06d}"
        current = version["value"]
        last = i+1 == plan["total_updates"]
        pilot_stop=False
        if pilot and not validation and not scale:
            update_record=read_json(directory/'update.json')
            pilot_stop=last or update_record['optimizer']['pilot_gradient']['reward_gradient_nonzero']
            last=pilot_stop
        global_completed = global_start+i+1
        dev_due = last or global_completed % plan["development_every_updates"] == 0
        if scale and pilot["stage"]=="capacity":dev_due=False
        save_due = validation or (scale and pilot["stage"]=="capacity") or dev_due or global_completed % plan["checkpoint_every_updates"] == 0
        def write_payload(staging):
            from .checkpoint_io import verified_local_copy,state_digest,rng_probe
            payload=dict(optimizer=optimizer.state_dict(),iteration=i,seed=seed,policy_version=current,
                binding=bound,cpu_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(actor_device),
                global_update_index=global_start+i)
            if scale:payload.update(cuda_rng_all=torch.cuda.get_rng_state_all(),query_schedule_position=i+1,query_schedule_sha256=fingerprint(frozen_schedule))
            expected_digest=state_digest(payload) if checkpoint_io=='local_verified_v1' else None
            def write_local(folder):
                actor.save_pretrained(folder/'actor',safe_serialization=True)
                tokenizer.save_pretrained(folder/'actor')
                # Python file descriptor catches flush/fsync errors before publication.
                with (folder/'optimizer-and-rng.pt').open('wb') as fd:
                    torch.save(payload,fd);fd.flush();os.fsync(fd.fileno())
                atomic_json(folder/'rng-probe.json',rng_probe(payload,actor_device))
            def verify_local(folder):
                model,info=AutoModelForCausalLM.from_pretrained(str(folder/'actor'),torch_dtype=torch.float32,
                    output_loading_info=True,**local)
                if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
                    raise ValueError('local checkpoint model load is incomplete')
                model.eval()
                if identify(model,behavior)[0]!=current:raise ValueError('local checkpoint differs from live policy')
                del model
                saved=torch.load(folder/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
                if state_digest(saved)!=expected_digest:raise ValueError('serialized optimizer or RNG differs from live snapshot')
            if checkpoint_io=='local_verified_v1':
                verified_local_copy(staging,write_local,verify_local,temporary_root=deployment.get('checkpoint_staging_root'))
            else:write_local(staging)
        if save_due:
            cp_path = directory/"checkpoint"
            if not recovered:
                publish_checkpoint(directory,iteration=i,policy_version=current,binding_sha256=bound,write=write_payload)
            checked = verify_checkpoint(cp_path,bound,allow_missing_stage=True)
            if checked["policy_version"] != current:
                raise ValueError("restored actor and saved checkpoint differ")
            if not recovered and not (scale and checkpoint_io=="local_verified_v1"):
                reloaded, info = AutoModelForCausalLM.from_pretrained(str(cp_path/"actor"),torch_dtype=torch.float32,
                                                                      output_loading_info=True,**local)
                if any(info.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")):
                    raise ValueError("checkpoint reload has incomplete weights")
                reloaded.eval()
                reload_version, _ = identify(reloaded,behavior)
                if reload_version != current:
                    raise ValueError("reloaded checkpoint weights differ from saved policy")
                del reloaded
                atomic_json(directory/"reload-verification.json",dict(policy_version=current,weights_verified=True))
            if scale and not recovered and checkpoint_io=="local_verified_v1":
                io=read_json(cp_path/"io-verification.json")
                if not io.get("local_payload_verified") or not io.get("shared_copy_verified"):
                    raise ValueError("checkpoint copy lacks verified local reload")
                atomic_json(directory/"reload-verification.json",dict(policy_version=current,weights_verified=True,verification_source="local Actor reload plus exact shared-copy hashes; duplicate reload omitted"))
            complete_stage(directory,"checkpoint_complete",current,{"checkpoint/complete.json":file_sha256(cp_path/"complete.json")})
        else:
            complete_stage(directory,"checkpoint_complete",current,{"skipped":"cadence"})
        if checkpoint_diagnostic_mode(deployment,plan,parent=parent,resume=resume):
            if validation and recovered:
                from .training_validation import validate_committed_progress
                record=read_json(directory/'training-validation.json')
                validate_committed_progress(record,index=i,policy=current,
                    checkpoint=file_sha256(directory/'checkpoint/complete.json'),risk=file_sha256(directory/'pilot-post-update-risk.json'))
                return record if last else None
            if pilot and not pilot_stop and not validation and not scale:
                complete_stage(directory,'development_complete',current,{'skipped':'bounded pilot; no development evaluation'})
                complete_stage(directory,'iteration_complete',current,{'selected_checkpoint_so_far':None})
                return None
            if scale and recovered and (root/"capacity-complete.json").exists():
                return read_json(root/"capacity-complete.json")
            # Check the real post-update engine snapshot on one fixed native
            # condition. This is a sampling probe, without retrieval or PPO.
            saved=read_json(directory/'trajectories.json')
            original=saved[0]
            if original['steps']:
                prefix=original['steps'][0]['prefix_token_ids']
            elif original.get('failure'):
                prefix=original['failure']['prefix_token_ids']
            else:
                raise ValueError('diagnostic original has no saved native action condition')
            with operation('post_update_sampling_probe',policy_version=current):
                generated=generator.generate(tuple(prefix),seed=stable_seed(seed,'post_update_probe',0))
            if generator.policy_version!=current:
                raise ValueError('post-update probe used a stale policy')
            atomic_json(directory/'post-update-sampling-probe.json',dict(policy_version=current,generation=generated))
            if pilot:
                from .precision_pilot import mismatch_statistics,risk_screen
                generator.prepare_scoring()
                ids=tuple(prefix)+generated.token_ids
                mask=(False,)*len(prefix)+generated.action_mask
                with torch.no_grad(),autocast():
                    teacher=sampled_old_log_probs(actor,ids,mask,device=actor_device,logit_chunk_size=deployment['logit_chunk_size'],
                        max_context_tokens=plan['max_context_tokens'],policy_call_kwargs={},projection_mode=deployment.get("hf_projection_mode","full_sequence"))[0].cpu().tolist()
                q=[v for v,m in zip(generated.sampled_log_probs,generated.action_mask,strict=True) if m]
                p=[v for v,m in zip(teacher,mask,strict=True) if m]
                stats=mismatch_statistics(q,p)
                outside_limit=.02 if validation and i>0 else .01
                passed=risk_screen(stats,outside_max=outside_limit)
                atomic_json(directory/'pilot-post-update-risk.json',dict(policy_version=current,statistics=stats,
                    fresh_worker_snapshot=True,outside_limit=outside_limit,pass_risk_screen=passed))
                if not passed:raise ValueError('updated pilot sampler/teacher risk screen failed')
            result=dict(status='diagnostic_update_checkpoint_and_sampling_complete',
                completed_updates=i+1,policy_version=current,checkpoint=str(directory.relative_to(root)/'checkpoint'),
                development='not_run_diagnostic_scope',checkpoint_selection='not_performed',
                optimizer_step_replayed=False)
            if validation:
                result.update(status='paired_validation_update_complete',
                    checkpoint_complete_sha256=file_sha256(directory/'checkpoint/complete.json'),
                    post_update_risk_sha256=file_sha256(directory/'pilot-post-update-risk.json'))
                complete_stage(directory,'development_complete',current,{'skipped':'explicit validation scope; no development score'})
                complete_stage(directory,'iteration_complete',current,{'validation_only':True,'selected_checkpoint_so_far':None})
                atomic_json(directory/'training-validation.json',result)
                if last:atomic_json(root/'training-validation-complete.json',result)
                return result
            if scale:
                result.update(status="capacity_update_complete",method=plan["method"],optimizer_steps=1,queries=32,original_trajectories=128)
                complete_stage(directory,"development_complete",current,{"skipped":"independent capacity batch, no development"})
                complete_stage(directory,"iteration_complete",current,{"capacity_complete":True})
                atomic_json(root/"capacity-complete.json",result)
                return result
            atomic_json(root/'diagnostic-complete.json',result)
            return result
        def evaluate(out):
            probe=[]
            with operation("development_evaluation",iteration=i,evaluation_scope=True):
                evaluate_queries(examples=dev,expected_ids=frozen.identities["development"],split="development",
                    phase="development",method=plan["method"],root_seed=seed,policy_version=current,
                    rollout_depth=plan["rollout_depth"],retrievers=retrievers,runners=runners,renderer=renderer,
                    reward=reward,output=out,scope="complete", manifest_sha256=frozen.manifest_sha256["development"],
                    backend_fingerprints={d:fingerprint(sources[d]) for d in retrievers},
                    samples_per_query=plan["eval_samples_per_query"],
                    trajectory_observer=(lambda t:probe.append(t) if not probe else None) if scale and last else None)
            if scale and last:
                t=probe[0]
                if t.policy_version!=current:raise ValueError('final evaluation sampled stale policy')
                generator.prepare_scoring()
                with operation('final_weight_probability_probe',iteration=i),torch.no_grad(),autocast():
                    scores=sampled_old_log_probs(actor,t.token_ids,t.generated_mask,device=actor_device,
                        logit_chunk_size=deployment['logit_chunk_size'],max_context_tokens=plan['max_context_tokens'],policy_call_kwargs={})[0].cpu().tolist()
                from .precision_pilot import mismatch_statistics,risk_screen
                q=trajectory_behavior_log_probs(t);m=t.generated_mask
                check=mismatch_statistics([x for x,b in zip(q,m,strict=True) if b],[x for x,b in zip(scores,m,strict=True) if b])
                passed=risk_screen(check,outside_max=.02)
                atomic_json(directory/'scale-final-weight-probe.json',dict(policy_version=current,
                    source='first fixed final evaluation query/sample; no extra generation',report=check,pass_risk_screen=passed))
                if not passed:raise ValueError('final refreshed-weight probability risk failed')
        finish_development(directory,current,due=dev_due,output=directory/"development-attempts"/attempt_dir.name,
                           evaluate=evaluate)
        best = selected_so_far(root,plan["checkpoint_selection"])
        complete_stage(directory,"iteration_complete",current,{"selected_checkpoint_so_far":best})
    start = 0
    if scale:
        from .scale_training import recomputation_generations
        recomputations=recomputation_generations(root)
    else:recomputations={}
    if restored:
        i = restored["iteration"]
        recovered_result=finish(i,recovered=True) if i>=0 else None
        if validation and recovered_result is not None:return recovered_result
        start = i+1
    for i in range(start,plan["total_updates"]):
        import time
        iteration_started=time.monotonic()
        with operation("training_iteration",iteration=i,recovery_generation=recomputations.get(i,0)):
            if completed_credit_recovery and i==0:
                completed_credit_recovery.run(components,iteration=i,policy_version=version['value'],output_root=root,
                    recipe=plan,branch_sink=sampled_branches)
            else:
                run_iteration(components,iteration=i,policy_version=version["value"],output_root=root,recipe=plan,root_seed=seed)
            diagnostic=finish(i)
        if scale:
            persisted=max((int(p.parent.parent.name[10:])+1 for p in root.glob('iteration-*/checkpoint/complete.json')),default=0)
            atomic_json(root/f'iteration-{i:06d}'/'scale-progress.json',dict(computed_updates=i+1,persisted_updates=persisted,
                iteration_seconds=time.monotonic()-iteration_started,policy_version=version['value'],queries=plan['queries_per_update'],
                checkpoint_complete=persisted==i+1,development_due=(pilot['stage']=='training' and (i+1==plan['total_updates'] or (i+1)%4==0)),
                recovery_generation=recomputations.get(i,0)))
            if pilot['stage']=='training' and persisted==i+1:
                from .scale_training import stop_at_checkpoint
                durations=[read_json(p)['iteration_seconds'] for p in sorted(root.glob('iteration-*/scale-progress.json'))]
                if stop_at_checkpoint(durations,i+1,plan['total_updates']):
                    result=dict(status='stopped_at_persistent_checkpoint',computed_updates=i+1,persisted_updates=persisted,
                        requested_updates=plan['total_updates'],policy_version=version['value'])
                    atomic_json(attempt_dir/'planned-stop.json',result)
                    return result
        if diagnostic is not None:
            atomic_json(attempt_dir/'cost-report.json',cost_report(root))
            return diagnostic
        if not pilot and plan["method"] == "ricepo" and plan["format_failure_policy"] == "invalid_action_terminal_zero_v1":
            evidence = read_json(root/f"iteration-{i:06d}"/"credit-diagnostics.json")
            if evidence["estimable_anchor_count"] == 0:
                result = dict(status="paused_no_estimable_rice_credit",completed_updates=i+1,
                              last_policy_version=version["value"],
                              reason="All selected anchors were not estimable; do not spend further RICE training budget.")
                atomic_json(root/"stopped.json",result)
                return result
    best = selected_so_far(root,plan["checkpoint_selection"])
    if not best:
        raise ValueError("no eligible checkpoint with complete development results")
    result = {"completed_updates":plan["total_updates"],"last_policy_version":version["value"],
              "selected_checkpoint":best,"output_root":str(root),"checkpoint_selection":plan["checkpoint_selection"]}
    if parent:
        previous=parent['lineage']['parent_selected_checkpoint']
        # Keep local selection for existing recovery; expose the full-lineage choice separately.
        winner = best if (plan['checkpoint_selection']=='last' or
                          best['development_macro']>previous['development_macro']) else previous
        result.update(global_completed_updates=global_start+plan['total_updates'],
                      lineage_selected_checkpoint=winner,continuation=parent['lineage'])
    atomic_json(attempt_dir/"cost-report.json",cost_report(root))
    atomic_json(root/"completed.json",result)
    return result

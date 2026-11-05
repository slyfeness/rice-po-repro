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
import random
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
from .contracts import Generation
from .data import QueryExample
from .formats import NativeTagParser, PromptRenderer
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
        native = tuple(output[0, prefix_length:].tolist())
        at_limit = len(native) >= limit
        engine_reason = ("eos" if native and native[-1] == tokenizer.eos_token_id
                         else "length_or_stop" if at_limit else "stopping_criterion")
        raw_completion(native, engine_reason, prefix, seed, version)
        if self.policy_version != version:
            raise RuntimeError("actor policy changed during a frozen-policy generation")
        if output.ndim != 2 or output.shape[0] != 1 or tuple(output[0, :prefix_length].tolist()) != prefix:
            raise ValueError("model.generate altered the native prefix or completion count")
        text = tokenizer.decode(native, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reason = "stop" if "</summary>" in text else engine_reason
        event("completion_diagnosis", finish_reason=reason, decoded_preview=text[:400])
        if reason != "stop" and at_limit:
            raise GenerationTruncated("generation exhausted its configured length before closing the summary")
        return self.parser(native, reason)


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
        dense = DenseVectorScorer(ids, vectors, encoder.query, chunk_size=deployment["dense_chunk_size"])
        sparse = LuceneBM25Scorer(corpus)
        backend = FixedHybridRetriever(dense_score=dense, sparse_score=sparse, corpus=corpus,
            normalization=minmax, exclude_before_normalization=settings["excluded_before_normalization"],
            top_k=deployment["retrieval_top_k"], render_observation=renderer.observation if renderer else lambda docs: (),
            backend_version=spec["backend_version"])
        audit.update(backend_version=spec["backend_version"], ranking=backend.recipe(),
                     query_composition=settings.get("query_composition", {"mode":"original_query_D0"}),
                     observation_top_k=settings.get("observation_top_k",0),
                     observation_token_limit=settings.get("observation_token_limit",0))
    return backend,audit


def run(config, deployment, *, resume=False):
    require_execution(config, command="train")
    require_slurm()
    if int(os.environ.get("WORLD_SIZE","1")) != 1 or deployment.get("runtime") != "single_process_hf_reference":
        raise ValueError("only the one-process reference backend is supported")
    root = require_cluster_storage(deployment["output_root"])
    with writer(root,resume=resume), attempt(root,recovery=resume) as attempt_dir:
        return _train(config,deployment,root,attempt_dir,resume)


def _train(config,deployment,root,attempt_dir,resume):
    from .assessment import evaluate_queries
    resolved = resolve(config,"train")
    plan = resolved["values"]
    with operation("asset_binding"):
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
        precision = deployment["dtype"]
        if precision not in {"float32","bfloat16"}:
            raise ValueError("unsupported precision")
        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16,enabled=precision=="bfloat16")
        local = dict(local_files_only=True,trust_remote_code=False)
        actor_source = local_model_identity(actor_path)
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
    binding = dict(schema_version=2,config_sha256=fingerprint(config),recipe_fingerprint=resolved["fingerprint"],
                   source_sha256=fingerprint(_source_hashes()),deployment_sha256=fingerprint(
                   {k:v for k,v in deployment.items() if k not in {"test_scope","expected_test_domains"}}),
                   dataset_receipt_sha256=frozen.receipt_sha256,manifest_sha256=frozen.manifest_sha256,
                   model_sha256=fingerprint(model_sources),retriever_sha256=fingerprint(sources),prompt_sha256=file_sha256(prompt_path),
                   behavior=dict(root_seed=seed, temperature=plan["temperature"],top_p=plan["top_p"],top_k=plan["top_k"],
                                 repetition_penalty=1.0,model_mode="eval", precision=precision,
                                 initial_actor_payload=actor_source["payload_sha256"],tokenizer_length=len(tokenizer),
                                 special_ids=tokenizer.all_special_ids,chat_template_sha256=fingerprint(tokenizer.chat_template),
                                 prefilled_think=True, rng="sha256-json-stream-v1"))
    bound = bind_run(root,binding,resume=resume)
    fixed_files = {"config.json":config,"deployment.json":deployment,"recipe.json":resolved,
                   "model-sources.json":model_sources,"retriever-sources.json":sources,"sources.json":_source_hashes()}
    for name,value in fixed_files.items():
        if resume:
            if read_json(root/name) != value:
                raise ValueError(f"changed run metadata: {name}")
        else:
            atomic_json(root/name,value)
    restored = None
    if resume:
        cp, restored = recovery_checkpoint(root,bound)
        if (root/"completed.json").exists():
            finished = read_json(root/"completed.json")
            verify_checkpoint(require_cluster_storage(finished["selected_checkpoint"]["checkpoint"]),bound)
            if (finished["selected_checkpoint"] != selected_so_far(root,plan["checkpoint_selection"])
                    or restored["iteration"]+1 != plan["total_updates"]):
                raise ValueError("completed run selection or update position differs")
            return finished
        actor_load = cp/"actor"
    else:
        actor_load = actor_path
    def load_actor(path):
        model, info = AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=torch.float32,
                                                           output_loading_info=True,**local)
        if any(info.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")):
            raise ValueError("partial actor weight load")
        return model.to(actor_device).eval()
    actor = load_actor(actor_load)
    reference = load_actor(actor_path)
    reference.requires_grad_(False)
    if local_model_identity(actor_path) != actor_source:
        raise ValueError("actor/tokenizer assets changed while loading")
    opt = plan["optimizer_hyperparameters"]
    optimizer = torch.optim.AdamW(actor.parameters(),lr=plan["actor_learning_rate"],betas=tuple(opt["betas"]),
                                  eps=opt["eps"],weight_decay=opt["weight_decay"])
    if restored:
        state = torch.load(cp/"optimizer-and-rng.pt",map_location="cpu",weights_only=True)
        if (state["binding"] != bound or state["iteration"] != restored["iteration"] or state["seed"] != seed
                or state["policy_version"] != restored["policy_version"]):
            raise ValueError("checkpoint optimizer identity disagrees with run binding")
        optimizer.load_state_dict(state["optimizer"])
        torch.random.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"],actor_device)
    behavior = binding["behavior"]
    def register_policy():
        key,evidence = identify(actor,behavior)
        path = root/"behavior-identities"/(key+".json")
        if path.exists():
            if read_json(path) != evidence:
                raise ValueError("behavior identity collision")
        else:
            atomic_json(path,evidence)
        return key
    version = {"value":register_policy()}
    if restored and version["value"] != restored["policy_version"]:
        raise ValueError("restored live policy differs from committed behavior identity")
    generator = HFNativeGenerator(model=actor,tokenizer=tokenizer,device=actor_device,
        policy_version_getter=lambda:version["value"],max_new_tokens=plan["max_new_tokens_per_step"],
        max_context_tokens=plan["max_context_tokens"],prefilled_think=True,precision_context=autocast)
    reward = PyTrecReward(qrels)
    runners = {domain:RolloutRunner(generator=generator,retriever=backend,reward_scorer=reward,
        query_composer=ExplicitQueryComposer(**plan["query_composition"]),rules=RolloutRules(
        frozenset({"stop"}),plan["max_new_tokens_per_step"],plan["max_context_tokens"],plan["rollout_depth"]),
        should_stop=lambda steps:False,final_reward=lambda steps:steps[-1].local_reward,
        excluded_ids_by_query={q:ex.excluded_ids for q,ex in examples.items() if ex.domain==domain})
        for domain,backend in retrievers.items()}
    scoring = make_forward(reference_model=reference,device=actor_device,logit_chunk_size=deployment["logit_chunk_size"],
                           max_context_tokens=plan["max_context_tokens"],policy_call_kwargs={},reference_call_kwargs={})
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
        selected = random.Random(stable_seed(seed,"query_sampling",i)).sample(sorted(train),plan["queries_per_update"])
        directory = root/f"iteration-{i:06d}"
        atomic_json(directory/"query-batch.json",selected)
        for q in selected:
            ex = train[q]
            with operation("retrieval_D0", query_id=q,iteration=i):
                initial = retrievers[ex.domain].retrieve(ex.query,excluded_ids=ex.excluded_ids)
                atomic_json(directory/"initial-evidence"/(fingerprint(q)+".json"),
                            {"query":q,"ranking":[asdict(d) for d in initial.documents]})
            prefix = renderer.initial(ex.query,initial.documents)
            for n in range(plan["trajectories_per_query"]):
                identity = f"{i}-{fingerprint(q)}-{n}"
                yield runners[ex.domain].run(query_id=q,trajectory_id=identity,original_query=ex.query,
                    initial_prefix_token_ids=prefix,old_policy_version=policy_version,
                    seeds=tuple(stable_seed(seed,"original",i,ex.domain,q,n,d) for d in range(1,plan["rollout_depth"]+1)))

    def collect_branches(t,a):
        ex = train[t.query_id]
        requests = [BranchRequest(f"{t.trajectory_id}-d{a.depth}-b{b}",
                    tuple(stable_seed(seed,"branch",t.policy_version,ex.domain,t.query_id,t.trajectory_id,a.depth,b,d)
                          for d in range(a.depth,plan["rollout_depth"]+1)))
                    for b in range(1,plan["total_branches_including_original"])]
        return runners[ex.domain].branch(original_query=ex.query,original=t,anchor=a,requests=requests)

    def update(originals,batch,old_version,directory):
        if version["value"] != old_version:
            raise ValueError("behavior policy changed before old log probabilities were cached")
        items = []
        with operation("old_log_probs"), torch.no_grad(), autocast():
            for t,adv,mask in zip(originals,batch.advantages,batch.generated_masks,strict=True):
                length = max(i for i,flag in enumerate(mask) if flag)
                with operation("old_probability_forward",input_tokens=length,model_passes=1,scored_tokens=sum(mask)):
                    logps = sampled_old_log_probs(actor,t.token_ids,mask,device=actor_device,
                        logit_chunk_size=deployment["logit_chunk_size"],max_context_tokens=plan["max_context_tokens"],
                        policy_call_kwargs={})
                items.append(UpdateItem(t.trajectory_id,old_version,t.token_ids,mask,adv,tuple(logps[0].cpu().tolist())))
        atomic_json(directory/"old-policy-update-items.json",items)
        with torch.enable_grad():
            diagnostics = ppo_plan_update(actor,optimizer,items,policy_version=old_version,forward=forward,
                                          device=actor_device,recipe=plan)
        version["value"] = register_policy()
        diagnostics["actual_state_dict_changed"] = version["value"] != old_version
        return {"new_policy_version":version["value"],"optimizer":diagnostics}

    def barrier(before,after):
        torch.cuda.synchronize(actor_device)
        torch.cuda.synchronize(retriever_device)

    components = IterationComponents(rollout_batch,entropy,collect_branches if entropy else None,update,barrier)
    def finish(i, *, recovered=False):
        directory = root/f"iteration-{i:06d}"
        current = version["value"]
        last = i+1 == plan["total_updates"]
        dev_due = last or (i+1) % plan["development_every_updates"] == 0
        save_due = dev_due or (i+1) % plan["checkpoint_every_updates"] == 0
        def write_payload(staging):
            actor.save_pretrained(staging/"actor",safe_serialization=True)
            tokenizer.save_pretrained(staging/"actor")
            torch.save(dict(optimizer=optimizer.state_dict(),iteration=i,seed=seed,policy_version=current,
                       binding=bound,cpu_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(actor_device)),
                       staging/"optimizer-and-rng.pt")
        if save_due:
            cp_path = directory/"checkpoint"
            if not recovered:
                publish_checkpoint(directory,iteration=i,policy_version=current,binding_sha256=bound,write=write_payload)
            checked = verify_checkpoint(cp_path,bound,allow_missing_stage=True)
            if checked["policy_version"] != current:
                raise ValueError("restored actor and saved checkpoint differ")
            complete_stage(directory,"checkpoint_complete",current,{"checkpoint/complete.json":file_sha256(cp_path/"complete.json")})
        else:
            complete_stage(directory,"checkpoint_complete",current,{"skipped":"cadence"})
        def evaluate(out):
            with operation("development_evaluation",iteration=i):
                evaluate_queries(examples=dev,expected_ids=frozen.identities["development"],split="development",
                    phase="development",method=plan["method"],root_seed=seed,policy_version=current,
                    rollout_depth=plan["rollout_depth"],retrievers=retrievers,runners=runners,renderer=renderer,
                    reward=reward,output=out,scope="complete", manifest_sha256=frozen.manifest_sha256["development"],
                    backend_fingerprints={d:fingerprint(sources[d]) for d in retrievers})
        finish_development(directory,current,due=dev_due,output=directory/"development-attempts"/attempt_dir.name,
                           evaluate=evaluate)
        best = selected_so_far(root,plan["checkpoint_selection"])
        complete_stage(directory,"iteration_complete",current,{"selected_checkpoint_so_far":best})
    start = 0
    if restored:
        i = restored["iteration"]
        finish(i,recovered=True)
        start = i+1
    for i in range(start,plan["total_updates"]):
        run_iteration(components,iteration=i,policy_version=version["value"],output_root=root,recipe=plan,root_seed=seed)
        finish(i)
    best = selected_so_far(root,plan["checkpoint_selection"])
    if not best:
        raise ValueError("no eligible checkpoint with complete development results")
    result = {"completed_updates":plan["total_updates"],"last_policy_version":version["value"],
              "selected_checkpoint":best,"output_root":str(root),"checkpoint_selection":plan["checkpoint_selection"]}
    atomic_json(attempt_dir/"cost-report.json",cost_report(root))
    atomic_json(root/"completed.json",result)
    return result

"""Pure command-specific expansion of the version-1 research recipe."""

from copy import deepcopy
from hashlib import sha256
import json
import math


TRAIN_METHODS = ("grpo", "ricepo")
BASELINES = ("retrieval_only", "prompting")
PROFILES = ("paper_reference", "smoke", "ablation")
CONFIG_FIELDS = {"schema_version", "profile", "method", "reference_plan", "reference_research_settings",
                 "implementation_provenance", "paper", "author_code", "execution", "paper_constants",
                 "batch_unit", "batch_unit_evidence", "model", "required_research_settings",
                 "setting_evidence", "deviations", "runtime_bindings"}
RUNTIME_BINDING_FIELDS = frozenset({"total_updates", "train_split_manifest", "development_split_manifest"})
PAPER_FIELDS = {
    "anchors_per_query": "anchor_top_k", "rollout_depth": "max_retrieval_depth",
    "influence_threshold": "influence_threshold", "residual_threshold": "residual_threshold",
    "actor_learning_rate": "actor_learning_rate", "actor_kl_coefficient": "actor_kl_coefficient",
    "metric": "metric", "reward_scale": "reward_scale",
}
LEGACY_NAMES = {"trajectories_per_query": "initial_trajectories_per_query"}
# Values declared in the independent reference choices; alternate values go through
# deviations so a child cannot silently change the reference against which it compares.
GROUP_REFERENCE = {
    "trajectories_per_query": 4, "total_branches_including_original": 6,
    "local_advantage_normalization": "center_only",
}
REFERENCE_PLAN = {
    "ppo_epochs": (1, "One pass over the fixed original trajectories collected per update."),
    "microbatch_size": (1, "Single-process reference forward/backward processes one trajectory at a time."),
    "max_grad_norm": (1.0, "Explicit gradient norm clipping convention for the independent optimizer."),
    "development_every_updates": (1, "Evaluate after each rollout update, and always after the final update."),
    "checkpoint_every_updates": (1, "Save each update, including the final update."),
}
ENCODE = ("document_encoding_profile",)
RETRIEVE = ENCODE + ("metric", "reward_scale", "hybrid_normalization", "excluded_before_normalization")
GENERATE = ("temperature", "top_p", "top_k", "max_new_tokens_per_step", "max_context_tokens",
            "training_seeds", "query_composition", "observation_top_k", "observation_token_limit",
            "prompt_artifact", "native_span_parser", "termination_policy", "format_failure_policy",
            "rollout_depth")
OPTIMIZE = ("queries_per_update", "trajectories_per_query", "trajectory_std_convention",
            "normalization_epsilon", "ppo_epochs", "optimizer_minibatch_size", "microbatch_size",
            "gradient_accumulation", "loss_reduction", "clip_epsilon", "kl_estimator", "actor_learning_rate",
            "actor_kl_coefficient", "optimizer", "optimizer_hyperparameters", "max_grad_norm", "total_updates",
            "train_split_manifest", "development_split_manifest", "checkpoint_selection",
            "development_every_updates", "checkpoint_every_updates")
CREDIT = ("anchors_per_query", "total_branches_including_original", "influence_threshold",
          "residual_threshold", "local_advantage_normalization", "anchor_selection", "gate_mode",
          "optimize_counterfactual_branches")
LEGACY_SETTINGS = {
    LEGACY_NAMES.get(k, k) for k in RETRIEVE + GENERATE + OPTIMIZE + CREDIT
    if k not in PAPER_FIELDS and k not in REFERENCE_PLAN and k not in (
        "queries_per_update", "optimizer_minibatch_size", "gradient_accumulation", "anchor_selection", "gate_mode")
}
DEPLOYMENT_BASE = ("runtime", "retriever_model_path", "cache_root", "retriever_device", "dtype",
                   "document_prefix", "retriever_max_length", "domains")
DEPLOYMENT_RETRIEVAL = ("dataset_receipt_path", "dataset_receipt_sha256", "train_manifest_path",
                        "development_manifest_path", "test_manifest_path", "dense_chunk_size", "retrieval_top_k")
DEPLOYMENT_GENERATION = ("actor_device", "prompt_path")
FIXED_BACKEND = {
    "temperature": (1,), "top_p": (1,), "top_k": (0,), "optimizer": ("adamw",),
    "kl_estimator": ("exact_categorical",), "trajectory_std_convention": ("population",),
    "hybrid_normalization": ("minmax",), "format_failure_policy": ("raise",),
    "native_span_parser": ("strict_native_tag_boundaries_prefilled_think",),
    "document_encoding_profile": ("original_model_card_raw_documents",),
    "loss_reduction": ("sum", "token_mean", "sequence_mean"),
    "local_advantage_normalization": ("center_only", "center_std"),
    "checkpoint_selection": ("last", "best_development_macro"),
    "anchor_selection": ("entropy", "random"),
    "gate_mode": ("paper", "influence_only", "residual_only", "always", "never"),
}


def _source(status, reason):
    return {"status": status, "reason": reason}


def _paper(config, locator):
    return {"status": "paper_explicit", "url": config.get("paper"), "locator": locator}


def _scope(method, command):
    if command == "compare-retrieval":
        fields = RETRIEVE + (() if method == "retrieval_only" else ("query_composition",))
        return fields, DEPLOYMENT_BASE + ("dense_chunk_size", "retrieval_top_k")
    if command == "encode-corpus":
        return ENCODE, DEPLOYMENT_BASE
    fields = RETRIEVE
    deployment = DEPLOYMENT_BASE + DEPLOYMENT_RETRIEVAL
    if method != "retrieval_only":
        fields += GENERATE
        deployment += DEPLOYMENT_GENERATION
        deployment += ("actor_model_path",)
    if command == "train":
        fields += OPTIMIZE
        deployment += ("output_root", "logit_chunk_size", "expected_development_domains")
        if method == "ricepo":
            fields += CREDIT
            deployment += ("entropy_chunk_size",)
    elif command == "evaluate":
        deployment += ("output_root", "expected_test_domains", "test_scope")
    elif command == "baseline":
        deployment += ("expected_test_domains", "test_scope")
    return fields, deployment


def _reference(config, constants, key):
    if key == "method":
        return config.get("method", "ricepo"), _source("implementation_choice", "Selected comparison method.")
    if key in PAPER_FIELDS:
        locator = {"metric": "Section 4.1 evaluation metric",
                   "reward_scale": "Section 4.2 retrieval reward"}.get(key, "Appendix A.1")
        return constants[PAPER_FIELDS[key]], _paper(config, locator)
    if key == "queries_per_update":
        declared = config.get("batch_unit") == "queries"
        return (constants["training_batch_size"] if declared else None,
                _source("implementation_choice", config.get("batch_unit_evidence")))
    if key in ("anchor_selection", "gate_mode"):
        return (("entropy", _paper(config, "Equation (2)")) if key == "anchor_selection"
                else ("paper", _paper(config, "Equation (6)")))
    if key in REFERENCE_PLAN:
        value, reason = REFERENCE_PLAN[key]
        return (value if config.get("reference_plan") == "single_process_v1" else None,
                _source("implementation_choice", reason))
    name = LEGACY_NAMES.get(key, key)
    declared = config.get("required_research_settings", {}).get(name)
    if declared is None and name in RUNTIME_BINDING_FIELDS:
        binding = config.get("runtime_bindings", {}).get(name)
        if binding is not None:
            return binding, _source("runtime_binding", "Explicit run-local binding for an otherwise unresolved field.")
    return declared, config.get("setting_evidence", {}).get(name, {})


def _positive_int(value):
    return type(value) is int and value > 0


def _finite(value):
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


def _derived(key, values, has_plan):
    if key == "optimizer_minibatch_size":
        queries, trajectories = values.get("queries_per_update"), values.get("trajectories_per_query")
        size = queries * trajectories if _positive_int(queries) and _positive_int(trajectories) else None
        return (size if has_plan else None,
                _source("derived", "All original trajectories in queries_per_update × trajectories_per_query."))
    minibatch, microbatch = values.get("optimizer_minibatch_size"), values.get("microbatch_size")
    count = None
    if _positive_int(minibatch) and _positive_int(microbatch) and minibatch % microbatch == 0:
        count = minibatch // microbatch
    return count if has_plan else None, _source("derived", "optimizer_minibatch_size / microbatch_size.")


def _validate_evidence(key, source, issues):
    status = source.get("status") if isinstance(source, dict) else None
    if status in ("implementation_choice", "experiment_override", "derived", "runtime_binding"):
        if not isinstance(source.get("reason"), str) or not source["reason"].strip():
            issues.append(f"undocumented implementation choice: {key}")
    elif status in ("paper_explicit", "author_code_explicit", "upstream_explicit"):
        if not source.get("url") or not source.get("locator"):
            issues.append(f"missing source locator: {key}")
    else:
        issues.append(f"unknown evidence status: {key}")


def _validate_values(values, issues):
    for key, allowed in FIXED_BACKEND.items():
        if key in values and values[key] is not None:
            if type(values[key]) is bool or values[key] not in allowed:
                issues.append(f"unsupported {key}: {values[key]!r}")
    for key in ("queries_per_update", "ppo_epochs", "optimizer_minibatch_size", "microbatch_size",
                "gradient_accumulation", "development_every_updates", "checkpoint_every_updates",
                "total_updates", "max_new_tokens_per_step", "max_context_tokens", "observation_top_k",
                "observation_token_limit", "rollout_depth"):
        if values.get(key) is not None and not _positive_int(values[key]):
            issues.append(f"{key} must be a positive integer")
    for key in ("trajectories_per_query", "total_branches_including_original"):
        val = values.get(key)
        if val is not None and (not _positive_int(val) or val < 2):
            issues.append(f"{key} must be an integer >= 2")
    for key in ("normalization_epsilon", "clip_epsilon", "max_grad_norm", "actor_learning_rate",
                "actor_kl_coefficient", "influence_threshold", "residual_threshold"):
        val = values.get(key)
        if val is not None:
            zero_allowed = key in ("influence_threshold", "residual_threshold", "actor_kl_coefficient")
            if not _finite(val) or val < 0 or (val == 0 and not zero_allowed):
                issues.append(f"{key} must be finite and {'nonnegative' if zero_allowed else 'positive'}")
    clip = values.get("clip_epsilon")
    if _finite(clip) and clip >= 1:
        issues.append("clip_epsilon must be < 1")
    for key in ("optimize_counterfactual_branches", "excluded_before_normalization"):
        if values.get(key) is not None and type(values[key]) is not bool:
            issues.append(f"{key} must be a boolean")
    if values.get("optimize_counterfactual_branches") is True:
        issues.append("optimizing counterfactual branches is unsupported")
    seed = values.get("training_seeds")
    if seed is not None and (not isinstance(seed, list) or len(seed) != 1 or type(seed[0]) is not int
                             or not 0 <= seed[0] < 2**63):
        issues.append("training_seeds must contain one integer in [0, 2**63)")
    composer = values.get("query_composition")
    if composer is not None and (not isinstance(composer, dict) or set(composer) != {"mode", "separator"}
            or composer["mode"] not in ("summary_only", "original_and_summary")
            or not isinstance(composer["separator"], str)):
        issues.append("query_composition requires an explicit supported mode and a string separator")
    for key in ("prompt_artifact", "train_split_manifest", "development_split_manifest"):
        if values.get(key) is not None and (not isinstance(values[key], str) or not values[key].strip()):
            issues.append(f"{key} must be a nonempty path string")
    options = values.get("optimizer_hyperparameters")
    if options is not None:
        if not isinstance(options, dict) or set(options) != {"betas", "eps", "weight_decay"}:
            issues.append("optimizer_hyperparameters must specify only AdamW betas, eps and weight_decay")
        else:
            betas = options["betas"]
            if (not isinstance(betas, list) or len(betas) != 2
                    or any(not _finite(b) or not 0 <= b < 1 for b in betas)
                    or not _finite(options["eps"]) or options["eps"] <= 0
                    or not _finite(options["weight_decay"]) or options["weight_decay"] < 0):
                issues.append("invalid optimizer_hyperparameters for AdamW")
    mini, micro, accum = (values.get(k) for k in ("optimizer_minibatch_size", "microbatch_size", "gradient_accumulation"))
    queries, trajectories = values.get("queries_per_update"), values.get("trajectories_per_query")
    anchors, depth = values.get("anchors_per_query"), values.get("rollout_depth")
    if anchors is not None and (type(anchors) is not int or anchors < 0):
        issues.append("anchors_per_query must be a nonnegative integer")
    elif (anchors is not None and _positive_int(trajectories) and _positive_int(depth)
          and anchors > trajectories * depth):
        issues.append("anchors_per_query exceeds the number of pooled eligible summary actions")
    if all(_positive_int(x) for x in (queries, trajectories, mini)) and queries * trajectories % mini:
        issues.append("optimizer_minibatch_size must evenly divide the original trajectory batch")
    if all(_positive_int(x) for x in (mini, micro, accum)) and mini != micro * accum:
        issues.append("optimizer_minibatch_size must equal microbatch_size × gradient_accumulation")
    if micro is not None and micro != 1:
        issues.append("the single-process reference supports only microbatch_size=1")
    dev, ckpt = values.get("development_every_updates"), values.get("checkpoint_every_updates")
    if _positive_int(dev) and _positive_int(ckpt) and dev % ckpt:
        issues.append("development_every_updates must be a multiple of checkpoint_every_updates")
    termination = values.get("termination_policy")
    if _positive_int(depth) and termination is not None and termination != f"full_depth_{depth}":
        issues.append("termination_policy must match the complete rollout_depth")


def _declared_changes(config, profile, fields, issues):
    changes = config.get("deviations", {})
    if not isinstance(changes, dict):
        issues.append("deviations must map field names to {value, reason} objects")
        return {}
    if profile == "paper_reference" and changes:
        issues.append("paper_reference does not allow deviations")
    if profile == "smoke" and set(changes) != {"queries_per_update"}:
        issues.append("smoke overrides only queries_per_update, from 128 to 1")
    if profile == "ablation" and not changes:
        issues.append("ablation requires at least one explicit deviation")
    parsed = {}
    for key, entry in changes.items():
        if key not in fields or key in ("method", "metric", "reward_scale", "total_updates",
                "train_split_manifest", "development_split_manifest", "termination_policy"):
            issues.append(f"unsupported deviation for this method: {key}")
        elif (not isinstance(entry, dict) or set(entry) != {"value", "reason"}
                or entry["value"] is None or not isinstance(entry["reason"], str) or not entry["reason"].strip()):
            issues.append(f"deviations.{key} requires a value and a nonempty reason")
        else:
            parsed[key] = entry
    if profile == "smoke" and parsed.get("queries_per_update", {}).get("value") != 1:
        issues.append("smoke requires queries_per_update=1")
    return parsed


def _runtime_bindings(config, issues):
    bindings = config.get("runtime_bindings", {})
    if not isinstance(bindings, dict):
        issues.append("runtime_bindings must be an object")
        return
    for key, value in bindings.items():
        if key not in RUNTIME_BINDING_FIELDS:
            issues.append(f"unsupported runtime binding: {key}")
        elif value is None or (isinstance(value, str) and not value.strip()):
            issues.append(f"runtime binding {key} must be nonempty")


def expand(config, command, constants, independent_settings=None):
    issues = []
    method = config.get("method", "ricepo")
    profile = config.get("profile", "paper_reference")
    if profile == "independent_reference_choices_not_author_recipe":
        profile = "paper_reference"
    if profile not in PROFILES:
        issues.append(f"unknown research profile: {profile!r}")
    if config.get("reference_plan") not in (None, "single_process_v1"):
        issues.append(f"unsupported reference_plan: {config['reference_plan']!r}")
    if method not in TRAIN_METHODS + BASELINES:
        issues.append(f"unsupported method: {method!r}")
    dispatch = "train" if method in TRAIN_METHODS else "baseline"
    if command == "inspect":
        command = dispatch
    if command not in ("train", "evaluate", "baseline", "encode-corpus", "compare-retrieval"):
        issues.append(f"unsupported recipe command: {command!r}")
    if command in ("train", "evaluate") and method not in TRAIN_METHODS:
        issues.append(f"{command} requires a trained method (ricepo or grpo)")
    if command == "baseline" and method not in BASELINES:
        issues.append("baseline requires retrieval_only or prompting")
    reported = config.get("paper_constants", {})
    if not isinstance(reported, dict):
        reported = {}
        issues.append("paper_constants must be an object")
    for key, expected in constants.items():
        if reported.get(key) != expected or type(reported.get(key)) is bool:
            issues.append(f"paper_constants.{key}: expected {expected!r}")
    for key in sorted(reported.keys() - constants.keys()):
        issues.append(f"unknown paper_constants.{key}")
    for key in ("required_research_settings", "setting_evidence"):
        if not isinstance(config.get(key, {}), dict):
            issues.append(f"{key} must be an object")
            config = {**config, key: {}}
        else:
            for unknown in sorted(config.get(key, {}).keys() - LEGACY_SETTINGS):
                issues.append(f"unknown {key}.{unknown}; canonical recipe changes belong in deviations")
    if independent_settings is not None and not isinstance(independent_settings, dict):
        issues.append("reference_research_settings must be a settings object")
        independent_settings = {}
    _runtime_bindings(config, issues)
    fields, deployment = _scope(method, command)
    method_fields, _ = _scope(method, dispatch)
    # A deviation can describe training while a holdout or encoding command uses
    # only a subset of the method. Inactive fields are never passed to callbacks.
    changes = _declared_changes(config, profile, method_fields, issues)
    misplaced = (set(RETRIEVE + GENERATE + OPTIMIZE + CREDIT) & config.keys())
    for key in sorted(misplaced):
        issues.append(f"unexpected top-level {key}; declare research deviations under deviations")
    for key in sorted(config.keys() - CONFIG_FIELDS - misplaced):
        issues.append(f"unknown top-level recipe field: {key}")
    if not isinstance(config.get("execution", {}), dict):
        issues.append("execution must be an object")
    values, sources, deviations = {}, {}, {}
    has_plan = config.get("reference_plan") == "single_process_v1"
    for key in ("method", *method_fields):
        derived = key in ("optimizer_minibatch_size", "gradient_accumulation")
        if derived:
            reference, source = _derived(key, values, has_plan)
        else:
            reference, source = _reference(config, constants, key)
        name = LEGACY_NAMES.get(key, key)
        frozen = (independent_settings or {}).get(name)
        if key in GROUP_REFERENCE and reference is not None:
            frozen = GROUP_REFERENCE[key]
        if (frozen is not None and reference is not None and (key in fields or key in changes)
                and (reference != frozen or type(reference) is bool and type(frozen) is not bool)):
            issues.append(f"{key} changes the independent reference; declare its value under deviations")
            reference = frozen
        value = reference
        if key in changes:
            entry = changes[key]
            if reference is None:
                issues.append(f"deviations.{key} lacks a resolved reference")
            if entry["value"] == reference:
                issues.append(f"deviations.{key} must differ from its reference")
            deviations[key] = {"reference": reference, **entry}
            value, source = entry["value"], _source("experiment_override", entry["reason"])
        values[key], sources[key] = value, source
    # The policy refers to completion at the configured horizon; expose its
    # effective name even though the older inherited spelling encodes depth 5.
    if ("rollout_depth" in changes and values.get("termination_policy") == "full_depth_5"
            and _positive_int(values.get("rollout_depth"))):
        values["termination_policy"] = f"full_depth_{values['rollout_depth']}"
        sources["termination_policy"] = _source("derived", "Complete the resolved rollout_depth.")
    active = ("method", *fields)
    values = {key: values.get(key) for key in active}
    sources = {key: sources.get(key, {}) for key in active}
    unresolved = sorted(key for key, value in values.items() if value is None)
    for key in active:
        if key in unresolved:
            previous = sources[key]
            reason = previous.get("reason") if isinstance(previous, dict) else None
            sources[key] = _source("unresolved", reason or "This command requires an explicitly supplied value.")
            issues.append(f"unresolved: {key}")
        else:
            _validate_evidence(key, sources[key], issues)
    _validate_values(values, issues)
    model = config.get("model", {})
    use_actor = command not in {"encode-corpus","compare-retrieval"} and method != "retrieval_only"
    roles = ("retriever", "actor") if use_actor else ("retriever",)
    model_keys = [f"{role}_{suffix}" for role in roles for suffix in ("id", "revision")]
    identities = {key: model.get(key) if isinstance(model, dict) else None for key in model_keys}
    for key, value in identities.items():
        if not isinstance(value, str) or not value.strip():
            unresolved.append(f"model.{key}")
            issues.append(f"unresolved: model.{key}")
    result = {"command": command, "profile": profile, "method": method,
              "values": values, "sources": sources, "paper_constants": dict(constants),
              "model": identities, "deviations": deviations,
              "required_deployment_fields": list(deployment)}
    # Derived optimizer dimensions can differ even when the sole declared
    # experimental change is Q. Keep these consequences separate from factors.
    ref_batch = constants["training_batch_size"] * GROUP_REFERENCE["trajectories_per_query"]
    result["derived_changes"] = {
        k: {"reference": ref_batch, "value": values[k], "reason": sources[k]["reason"]}
        for k in ("optimizer_minibatch_size","gradient_accumulation")
        if k in values and values[k] is not None and values[k] != ref_batch
        and sources[k].get("status") == "derived"
    }
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    result.update(fingerprint=sha256(encoded).hexdigest(), unresolved_fields=sorted(unresolved),
                  issues=issues, execution=config.get("execution", {}))
    return deepcopy(result)

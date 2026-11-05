"""Fail-closed configuration: no undocumented science defaults."""

import json
import math
from pathlib import Path


class FidelityError(ValueError):
    pass


PAPER_CONSTANTS = {
    "anchor_top_k": 4, "max_retrieval_depth": 5,
    "influence_threshold": 0.05, "residual_threshold": 0.03,
    "actor_learning_rate": 1e-6, "actor_kl_coefficient": 0.001,
    "training_batch_size": 128, "metric": "ndcg_cut_10", "reward_scale": "0_to_1",
}

REQUIRED_SETTINGS = (
    "initial_trajectories_per_query", "total_branches_including_original",
    "local_advantage_normalization", "trajectory_std_convention",
    "normalization_epsilon", "optimize_counterfactual_branches", "temperature",
    "top_p", "top_k", "max_new_tokens_per_step", "max_context_tokens",
    "clip_epsilon", "loss_reduction", "kl_estimator", "optimizer",
    "optimizer_hyperparameters", "total_updates", "training_seeds",
    "query_composition", "hybrid_normalization", "excluded_before_normalization",
    "observation_top_k", "observation_token_limit", "document_encoding_profile",
    "prompt_artifact", "native_span_parser", "termination_policy",
    "format_failure_policy", "train_split_manifest", "development_split_manifest",
    "checkpoint_selection",
)


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise FidelityError("cyclic config inheritance")
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FidelityError("unsupported config schema")
    parent = value.pop("extends", None)
    if parent is not None:
        base = load_config(path.parent / parent, _seen | {path})
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **item}
            else:
                base[key] = item
        value = base
    return value


def fidelity_issues(config: dict, *, strict_author_source: bool = False) -> list[str]:
    issues = []
    constants = config.get("paper_constants", {})
    for key, expected in PAPER_CONSTANTS.items():
        if constants.get(key) != expected:
            issues.append(f"paper_constants.{key}: expected {expected!r}")
    author = config.get("author_code", {})
    if strict_author_source and (author.get("status") != "verified"
                                or not author.get("repository") or not author.get("commit")):
        issues.append("RICE-PO author repository/commit is not verified")
    settings = config.get("required_research_settings", {})
    evidence = config.get("setting_evidence", {})
    for key in REQUIRED_SETTINGS:
        if key not in settings or settings[key] is None:
            issues.append(f"unresolved: {key}")
            continue
        source = evidence.get(key, {})
        if source.get("status") == "implementation_choice":
            if not source.get("reason"):
                issues.append(f"undocumented implementation choice: {key}")
        elif source.get("status") not in {"paper_explicit", "author_code_explicit", "upstream_explicit"}:
            issues.append(f"unknown evidence status: {key}")
        elif not source.get("url") or not source.get("locator"):
            issues.append(f"missing source locator: {key}")
        if strict_author_source and source.get("status") not in {"paper_explicit", "author_code_explicit"}:
            issues.append(f"not author-verified: {key}")
    for key in ("initial_trajectories_per_query", "total_branches_including_original"):
        value = settings.get(key)
        if value is not None and (type(value) is not int or value < 2):
            issues.append(f"{key} must be an integer >= 2")
    for key in ("max_new_tokens_per_step", "max_context_tokens", "total_updates",
                "observation_top_k", "observation_token_limit"):
        value = settings.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            issues.append(f"{key} must be a positive integer")
    for key in ("normalization_epsilon", "temperature", "top_p", "clip_epsilon"):
        value = settings.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            issues.append(f"{key} must be positive and finite")
    for key, allowed in {
        "local_advantage_normalization": {"center_only", "center_std"},
        "trajectory_std_convention": {"population"},
        "loss_reduction": {"sum", "token_mean", "sequence_mean"},
    }.items():
        value = settings.get(key)
        if value is not None and value not in allowed:
            issues.append(f"unsupported {key}: {value}")
    if isinstance(settings.get("top_p"), (int, float)) and settings["top_p"] > 1:
        issues.append("top_p must be <= 1")
    if isinstance(settings.get("clip_epsilon"), (int, float)) and settings["clip_epsilon"] >= 1:
        issues.append("clip_epsilon must be < 1")
    for key in ("optimize_counterfactual_branches", "excluded_before_normalization"):
        if settings.get(key) is not None and type(settings[key]) is not bool:
            issues.append(f"{key} must be a boolean")
    return issues


def require_fidelity(config: dict) -> None:
    issues = fidelity_issues(config)
    if issues:
        raise FidelityError("Reproduction configuration is incomplete:\n- " + "\n- ".join(issues))


def require_execution(config: dict) -> None:
    require_fidelity(config)
    require_enabled(config)


def require_enabled(config: dict) -> None:
    execution = config.get("execution", {})
    if execution.get("enabled") is not True:
        raise FidelityError("execution is disabled by the current task scope")
    # runtime_validated is a recorded result, not a switch to set by assertion.
    # Enabling the first authorized smoke run must not require claiming it passed.

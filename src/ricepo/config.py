"""Read local recipe JSON and expand only the settings consumed by a command."""

import json
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

from .recipe import LEGACY_SETTINGS, expand


class FidelityError(ValueError):
    pass


PAPER_CONSTANTS = MappingProxyType({
    "anchor_top_k": 4, "max_retrieval_depth": 5,
    "influence_threshold": 0.05, "residual_threshold": 0.03,
    "actor_learning_rate": 1e-6, "actor_kl_coefficient": 0.001,
    "training_batch_size": 128, "metric": "ndcg_cut_10", "reward_scale": "0_to_1",
})
# Vocabulary accepted by version-1 configuration files. resolve() determines
# the subset that a particular command actually requires.
REQUIRED_SETTINGS = tuple(sorted(LEGACY_SETTINGS))


def _reject_nonfinite(literal):
    raise FidelityError(f"nonfinite JSON number in recipe: {literal}")


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise FidelityError("cyclic config inheritance")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise FidelityError("unsupported config schema")
    parent = value.pop("extends", None)
    if parent is not None and "reference_research_settings" in value:
        raise FidelityError("reference_research_settings is derived from the inherited independent reference")
    if parent is not None:
        if not isinstance(parent, str) or not parent.strip():
            raise FidelityError("extends requires a parent config path")
        base = load_config(path.parent / parent, _seen | {path})
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **item}
            else:
                base[key] = item
        value = base
    if (value.get("profile") == "independent_reference_choices_not_author_recipe"
            and value.get("reference_plan") == "single_process_v1"
            and "reference_research_settings" not in value):
        # Preserve the reference's settings before applying any child overrides.
        # Values and evidence used by a command are emitted by resolve().
        value["reference_research_settings"] = deepcopy(value.get("required_research_settings", {}))
    return value


def resolve(config: dict, command: str = "train") -> dict:
    """Return a static, JSON-serializable recipe report, even when inputs are missing.

    values contains the canonical field names passed to the command's callbacks.
    fingerprint identifies this command's plan, not model weights or dataset contents.
    Resolution reads no experiment assets and never authorizes execution.
    """
    return expand(config, command, PAPER_CONSTANTS, config.get("reference_research_settings"))


def fidelity_issues(config: dict, *, command: str = "train", strict_author_source: bool = False) -> list[str]:
    report = resolve(config, command)
    issues = list(report["issues"])
    if strict_author_source:
        author = config.get("author_code", {})
        if (author.get("status") != "verified" or not author.get("repository") or not author.get("commit")):
            issues.append("RICE-PO author repository/commit is not verified")
        for key, evidence in report["sources"].items():
            if evidence.get("status") not in {"paper_explicit", "author_code_explicit"}:
                issues.append(f"not author-verified: {key}")
    return issues


def require_fidelity(config: dict, command: str = "train") -> None:
    issues = fidelity_issues(config, command=command)
    if issues:
        raise FidelityError("Reproduction configuration is incomplete:\n- " + "\n- ".join(issues))


def require_execution(config: dict, command: str = "train") -> None:
    require_fidelity(config, command)
    require_enabled(config)


def require_enabled(config: dict) -> None:
    execution = config.get("execution", {})
    if execution.get("enabled") is not True:
        raise FidelityError("execution is disabled by the current task scope")
    # Validation flags are recorded outcomes. They are not permissions.

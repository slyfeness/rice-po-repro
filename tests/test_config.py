"""Configuration lifecycle contracts. Written only; never starts a runtime."""

import json
from pathlib import Path

import pytest

from ricepo.config import (
    FidelityError,
    fidelity_issues,
    load_config,
    require_execution,
    require_fidelity,
)


ROOT = Path(__file__).resolve().parents[1]


def configured_recipe():
    """Resolve a source fixture and supply the three explicitly absent inputs."""
    config = load_config(ROOT / "configs" / "implementation_choices.json")
    settings = config["required_research_settings"]
    settings["total_updates"] = 1
    settings["train_split_manifest"] = "fixture-train-manifest.jsonl"
    settings["development_split_manifest"] = "fixture-development-manifest.jsonl"
    return config


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_inheritance_resolves_relative_to_config_and_keeps_sibling_fields(tmp_path):
    write_json(tmp_path / "parent.json", {
        "schema_version": 1,
        "execution": {"enabled": False, "runtime_validated": False},
        "required_research_settings": {"temperature": 1.0, "total_updates": None},
    })
    child_dir = tmp_path / "profiles"
    child_dir.mkdir()
    write_json(child_dir / "child.json", {
        "schema_version": 1,
        "extends": "../parent.json",
        "required_research_settings": {"temperature": 0.8},
    })
    result = load_config(child_dir / "child.json")
    assert result["required_research_settings"] == {
        "temperature": 0.8, "total_updates": None,
    }
    assert result["execution"] == {"enabled": False, "runtime_validated": False}
    assert "extends" not in result
    assert json.loads((tmp_path / "parent.json").read_text(encoding="utf-8"))[
        "required_research_settings"
    ]["temperature"] == 1.0


def test_cyclic_config_inheritance_is_rejected(tmp_path):
    write_json(tmp_path / "a.json", {"schema_version": 1, "extends": "b.json"})
    write_json(tmp_path / "b.json", {"schema_version": 1, "extends": "a.json"})
    with pytest.raises(FidelityError, match="cyclic"):
        load_config(tmp_path / "a.json")


def test_reviewable_reference_choices_keep_budget_and_data_manifests_unresolved():
    config = load_config(ROOT / "configs" / "implementation_choices.json")
    issues = fidelity_issues(config)
    for field in ("total_updates", "train_split_manifest", "development_split_manifest"):
        assert f"unresolved: {field}" in issues
    assert config["execution"]["enabled"] is False
    assert not any("author repository/commit" in issue for issue in issues)


def test_independent_paper_implementation_does_not_require_author_repository():
    config = configured_recipe()
    config["author_code"] = {"status": "unverified", "repository": None, "commit": None}
    require_fidelity(config)
    assert fidelity_issues(config) == []
    assert "RICE-PO author repository/commit is not verified" in fidelity_issues(
        config, strict_author_source=True
    )


def test_implementation_choice_requires_a_reviewable_reason():
    config = configured_recipe()
    config["setting_evidence"]["temperature"] = {"status": "implementation_choice"}
    assert "undocumented implementation choice: temperature" in fidelity_issues(config)


def test_unresolved_settings_fail_before_execution_can_be_enabled():
    config = configured_recipe()
    config["required_research_settings"]["total_updates"] = None
    config["execution"]["enabled"] = True
    with pytest.raises(FidelityError, match="unresolved: total_updates"):
        require_execution(config)


def test_complete_recipe_still_cannot_execute_when_execution_is_disabled():
    config = configured_recipe()
    require_fidelity(config)
    with pytest.raises(FidelityError, match="execution is disabled"):
        require_execution(config)


def test_first_authorized_validation_does_not_require_claiming_runtime_already_passed():
    config = configured_recipe()
    config["execution"] = {
        "enabled": True, "runtime_validated": False, "four_gpu_validated": False,
    }
    # This validates only configuration. It creates no engine, process or job.
    require_execution(config)
    assert config["execution"]["runtime_validated"] is False
    assert config["execution"]["four_gpu_validated"] is False


def test_paper_variance_thresholds_cannot_silently_change():
    config = configured_recipe()
    config["paper_constants"]["influence_threshold"] = 5.0
    assert "paper_constants.influence_threshold: expected 0.05" in fidelity_issues(config)

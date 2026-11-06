"""Source fixtures for recipe expansion. These tests do not construct runtimes."""

from copy import deepcopy
from pathlib import Path

import pytest

from ricepo.config import FidelityError, load_config, require_execution, resolve


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def configured(name):
    return load_config(CONFIGS / f"{name}.json")


def test_smoke_changes_one_knob_and_recomputes_sizes_in_trajectories():
    reference = resolve(configured("paper_reference"), "train")
    smoke = resolve(configured("smoke"), "train")
    assert reference["values"]["queries_per_update"] == 128
    assert smoke["values"]["queries_per_update"] == 1
    assert smoke["paper_constants"]["training_batch_size"] == 128
    assert set(smoke["deviations"]) == {"queries_per_update"}
    assert reference["values"]["optimizer_minibatch_size"] == 512
    assert smoke["values"]["optimizer_minibatch_size"] == 4
    assert smoke["values"]["gradient_accumulation"] == 4
    assert smoke["values"]["trajectories_per_query"] == 4
    assert smoke["execution"] == dict(enabled=False, runtime_validated=False, four_gpu_validated=False)
    assert set(smoke["unresolved_fields"]) == {
        "total_updates", "train_split_manifest", "development_split_manifest",
    }


def test_expansion_is_repeatable_without_mutating_the_original_config():
    config = configured("ablation")
    original = deepcopy(config)
    a = resolve(config, "train")
    assert a["values"]["total_branches_including_original"] == 5
    assert a["deviations"]["total_branches_including_original"]["reference"] == 6
    assert a["sources"]["total_branches_including_original"]["status"] == "experiment_override"
    assert a["fingerprint"] == resolve(dict(reversed(list(config.items()))), "train")["fingerprint"]
    assert original == config


@pytest.mark.parametrize("profile", ["smoke", "paper_reference"])
def test_other_profiles_cannot_authorize_an_independent_ablation(profile):
    config = configured(profile)
    config.setdefault("deviations", {})["influence_threshold"] = {
        "value": 0.12, "reason": "Fixture sensitivity study.",
    }
    with pytest.raises(FidelityError, match=profile):
        require_execution(config)


def test_mutating_inherited_branch_count_cannot_redefine_reference_for_an_ablation():
    config = configured("ablation")
    config["required_research_settings"]["total_branches_including_original"] = 8
    assert any("total_branches_including_original" in issue for issue in resolve(config)["issues"]
               if not issue.startswith("unresolved:"))


@pytest.mark.parametrize("field,value", [
    ("gate_mode", "any"), ("influence_threshold", True), ("ppo_epochs", 0),
    ("optimizer_minibatch_size", 3), ("gradient_accumulation", 7),
    ("development_every_updates", 3), ("anchors_per_query", 21),
])
def test_invalid_credit_ppo_and_cadence_fields_are_rejected_before_runtime(field, value):
    config = configured("ablation")
    changes = {field: {"value": value, "reason": "Fixture change."}}
    if field == "development_every_updates":
        changes["checkpoint_every_updates"] = {"value": 2, "reason": "Fixture change."}
    config["deviations"] = changes
    with pytest.raises(FidelityError, match=field):
        require_execution(config)


@pytest.mark.parametrize("method", ["retrieval_only", "prompting"])
def test_evaluation_only_methods_do_not_consume_dummy_ppo_or_credit_plan(method):
    config = configured(method)
    config["required_research_settings"].pop("optimizer", None)
    config["required_research_settings"].pop("total_branches_including_original", None)
    result = resolve(config, "baseline")
    assert result["issues"] == []
    assert set(result["sources"]) == set(result["values"])
    assert {"optimizer", "anchors_per_query", "total_updates", "total_branches_including_original",
            "queries_per_update", "local_advantage_normalization"}.isdisjoint(result["values"])
    assert ("rollout_depth" in result["values"]) == (method == "prompting")
    assert ("actor_model_path" in result["required_deployment_fields"]) == (method == "prompting")
    with pytest.raises(FidelityError, match="train"):
        require_execution(config, "train")


def test_grpo_uses_group_normalization_and_ppo_without_branch_inputs():
    result = resolve(configured("grpo"), "train")
    assert result["values"]["trajectories_per_query"] == 4
    assert "optimizer" in result["values"]
    assert "total_branches_including_original" not in result["values"]
    assert "gate_mode" not in result["values"]
    assert "normalization_epsilon" in result["values"]


def test_holdout_and_corpus_encoding_do_not_require_a_training_budget():
    config = configured("paper_reference")
    holdout = resolve(config, "evaluate")
    encode = resolve(config, "encode-corpus")
    assert "total_updates" not in holdout["values"]
    assert "optimizer" not in holdout["values"]
    assert "prompt_artifact" in holdout["values"]
    assert "document_encoding_profile" in encode["values"]
    assert "prompt_artifact" not in encode["values"]
    assert encode["issues"] == []


def test_generated_evaluation_plan_declares_its_fixed_samples_per_query():
    plan = resolve(configured("grpo"), "train")
    assert plan["values"]["eval_samples_per_query"] == 1
    assert plan["sources"]["eval_samples_per_query"]["status"] == "implementation_choice"
    assert "eval_samples_per_query" not in resolve(configured("retrieval_only"), "baseline")["values"]


def test_scale32_original_and_summary_configs_keep_shared_rollout_budget():
    rice = resolve(configured("biology_scale32_ricepo"), "train")
    grpo = resolve(configured("biology_scale32_grpo"), "train")
    for plan in (rice, grpo):
        assert plan["values"]["queries_per_update"] == 32
        assert plan["values"]["trajectories_per_query"] == 4
        assert plan["values"]["optimizer_minibatch_size"] == 128
        assert plan["values"]["query_composition"] == {"mode": "original_and_summary", "separator": "\n"}
        assert plan["values"]["total_updates"] == 1
        assert plan["execution"]["enabled"] is False
    assert rice["values"]["total_branches_including_original"] == 6
    assert "total_branches_including_original" not in grpo["values"]

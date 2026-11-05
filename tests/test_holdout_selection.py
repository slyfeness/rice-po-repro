"""A held-out evaluation must respect the frozen development decision."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ricepo import holdout
from ricepo.artifacts import atomic_json
from ricepo.audit import fingerprint
from ricepo.config import load_config, resolve
from ricepo.state import selected_so_far


def test_formal_holdout_rejects_substitution_of_a_lower_scoring_checkpoint(tmp_path, monkeypatch):
    # Only the cluster storage path boundary is replaced to use this isolated local fixture.
    monkeypatch.setattr(holdout, "require_cluster_storage", lambda path: Path(path).resolve())
    root = tmp_path / "run"
    deployment = {"output_root": str(root)}
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "paper_reference.json")
    settings = config["required_research_settings"]
    settings.update(total_updates=2, train_split_manifest="train.jsonl", development_split_manifest="dev.jsonl")
    frozen = SimpleNamespace(receipt_sha256="a" * 64, manifest_sha256={"test": "b" * 64,"development":"d"*64})
    atomic_json(root / "config.json", config)
    atomic_json(root / "recipe.json", resolve(config))
    atomic_json(root / "sources.json", holdout._source_hashes())
    atomic_json(root / "deployment.json", deployment)
    atomic_json(root / "binding.json", dict(
        schema_version=2, config_sha256=fingerprint(config), recipe_fingerprint=resolve(config)["fingerprint"],
        source_sha256=fingerprint(holdout._source_hashes()), deployment_sha256=fingerprint(deployment),
        dataset_receipt_sha256=frozen.receipt_sha256, manifest_sha256=frozen.manifest_sha256,
    ))
    for i, dev_score in enumerate((0.25, 0.5)):
        directory = root / f"iteration-{i:06d}"
        digest = atomic_json(directory / "dev.json", {"macro": dev_score,"scope":"complete","split":"development",
            "policy_version":f"actor-{i}","manifest_sha256":"d"*64,"expected_total":1,"evaluated_total":1})
        atomic_json(directory / "development_complete.receipt.json", {
            "policy_version": f"actor-{i}", "artifacts": {"relative_report": "dev.json", "sha256": digest}})
    best = selected_so_far(root, "best_development_macro")
    substituted = dict(best, iteration=0, checkpoint=str(root / "iteration-000000" / "checkpoint"),
                       policy_version="actor-0", development_macro=0.25)
    atomic_json(root / "completed.json", {"selected_checkpoint": substituted})
    with pytest.raises(ValueError, match="development selection"):
        holdout._selected_state(config, deployment, substituted["checkpoint"], frozen)

"""Slurm CPU audit of one saved depth-two failure and its frozen D0 ranking."""
import argparse
from pathlib import Path

from ricepo.artifacts import atomic_json, require_cluster_storage, require_slurm
from ricepo.audit import fingerprint
from ricepo.dataset_bundle import verify_bundle
from ricepo.formats import action_closed
from ricepo.hf_runtime import _manifest
from ricepo.retrieval import ExplicitQueryComposer, PyTrecReward
from ricepo.state import read_json


def audit(run, recovered_run, output):
    require_slurm()
    from transformers import AutoTokenizer
    dep = read_json(run / "deployment.json")
    binding = read_json(run / "binding.json")
    bundle = verify_bundle(dep)
    if binding["manifest_sha256"] != bundle.manifest_sha256:
        raise ValueError("source manifest identities differ from the bundle")
    plan = read_json(run / "recipe.json")["values"]
    _, examples, qrels = _manifest(bundle.paths["train"], set(dep["domains"]),
                                   expected_sha256=bundle.manifest_sha256["train"])
    first, second = sorted(
        (row for path in run.glob("attempts/*/events/*.json")
         if (row := read_json(path)).get("kind") == "raw_completion"
         and row["phase"] == "original_generation"), key=lambda row: row["depth"])
    if ([first["depth"], second["depth"]] != [1, 2] or
        any(first[key] != second[key] for key in ("trajectory_id", "query_id", "policy_version"))):
        raise ValueError("unexpected source request pair")
    prefix = first["prefix_token_ids"] + first["token_ids"]
    if (second["prefix_token_ids"][:len(prefix)] != prefix or
        fingerprint(second["prefix_token_ids"]) != second["prefix_sha256"]):
        raise ValueError("round two does not extend the native round-one completion")
    tok = AutoTokenizer.from_pretrained(dep["actor_model_path"], local_files_only=True)
    if fingerprint(tok.chat_template) != binding["behavior"]["chat_template_sha256"]:
        raise ValueError("tokenizer template changed")
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    observation = decode(second["prefix_token_ids"][len(prefix):])
    q = first["query_id"]
    d0 = read_json(run / "iteration-000000/initial-evidence" / (fingerprint(q) + ".json"))
    ranking = [doc["document_id"] for doc in d0["ranking"]]
    corpus_ids = set(read_json(Path(dep["domains"][examples[q].domain]["vector_document_ids_json"])))
    positive = [doc for doc, grade in qrels[q].items() if grade > 0]
    if not set(positive) <= corpus_ids or set(ranking) & examples[q].excluded_ids:
        raise ValueError("corpus or excluded-id mismatch")
    reward = PyTrecReward(qrels)
    ideal = sorted(positive, key=lambda doc: (-qrels[q][doc], doc))
    control = reward.score(q, ideal)
    if abs(control - 1) > 1e-8:
        raise ValueError("positive-ranking NDCG control failed")
    iteration = recovered_run / "iteration-000000"
    provenance = read_json(recovered_run / "recovery-provenance.json")
    if provenance["dataset_receipt_sha256"] != bundle.receipt_sha256 or (
            provenance["retriever_identity"] != binding["retriever_sha256"]):
        raise ValueError("v5 recovery uses different assets")
    traces = read_json(iteration / "trajectories.json")
    for group in read_json(iteration / "counterfactual-trajectories.json"):
        traces.extend(group["trajectories"])
    composer = ExplicitQueryComposer(**plan["query_composition"])
    followups = []
    for trace in traces:
        if trace["query_id"] != q:
            raise ValueError("unexpected recovery query")
        for step in trace["steps"]:
            ranking_ids = [d["document_id"] for d in step["retrieval"]["documents"]]
            query = composer.compose(examples[q].query, step["generation"]["summary_text"])
            score = reward.score(q, ranking_ids)
            if (query != step["retrieval"]["query"] or score != step["local_reward"]
                    or not set(ranking_ids) <= corpus_ids or set(ranking_ids) & examples[q].excluded_ids):
                raise ValueError("retrieval query, ranking or reward mismatch")
            followups.append(dict(query_sha256=fingerprint(query), ndcg=score,
                                  positive_hits=len(set(ranking_ids[:10]) & set(positive))))
    atomic_json(output, dict(query_sha256=fingerprint(q), prefix_sha256=second["prefix_sha256"],
        first_completion_sha256=fingerprint(first["token_ids"]),
        previous_eos=first["token_ids"][-1] == tok.eos_token_id,
        inline_tool_observation=observation.startswith("\n<tool_response>"),
        assistant_prefill_count=observation.count("<think>"),
        round_one_has_action_close=action_closed(decode(first["token_ids"])),
        round_two_tag_counts={tag: decode(second["token_ids"]).count(tag)
                              for tag in ("<think>", "</think>", "<summary>", "</summary>")},
        d0_ndcg=reward.score(q, ranking), positive_id_count=len(positive),
        ideal_control_ndcg=control, recovered_followups=followups,
        query_composition=plan["query_composition"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run")
    parser.add_argument("v5_recovery")
    parser.add_argument("new_report")
    args = parser.parse_args()
    audit(require_cluster_storage(args.source_run), require_cluster_storage(args.v5_recovery),
          require_cluster_storage(args.new_report))

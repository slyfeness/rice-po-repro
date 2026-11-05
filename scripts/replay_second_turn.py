"""Replay exactly one saved second-turn seed after changing the observation framing."""
import argparse
from pathlib import Path

from ricepo.artifacts import atomic_json, require_cluster_storage, require_slurm
from ricepo.audit import attempt, fingerprint, operation
from ricepo.formats import CHAT_TURN, NativeTagParser, PromptRenderer, action_closed
from ricepo.hf_runtime import HFNativeGenerator
from ricepo.policy_identity import identify
from ricepo.state import read_json, writer


def replay(source, root):
    require_slurm()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    deployment, plan = read_json(source/"deployment.json"), read_json(source/"recipe.json")["values"]
    with writer(root, resume=False), attempt(root):
        events = [read_json(path) for path in source.glob("attempts/*/events/*.json")]
        first, second = sorted((e for e in events if e["kind"] == "raw_completion"
                                and e["phase"] == "original_generation"), key=lambda e: e["depth"])
        prefix = first["prefix_token_ids"] + first["token_ids"]
        if ([first["depth"], second["depth"]] != [1, 2] or
            any(first[k] != second[k] for k in ("query_id", "trajectory_id", "policy_version")) or
            prefix != second["prefix_token_ids"][:len(prefix)] or
            fingerprint(second["prefix_token_ids"]) != second["prefix_sha256"]):
            raise ValueError("saved two-round request identity mismatch")
        local = dict(local_files_only=True, trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(deployment["actor_model_path"], **local)
        def decode(ids):
            return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        obs = decode(second["prefix_token_ids"][len(prefix):])
        if not obs.startswith("\n<tool_response>") or not obs.endswith("</tool_response>\n<think>\n"):
            raise ValueError("unexpected historical observation")
        body = obs[1:-len("\n<think>\n")]
        renderer = PromptRenderer(tok, Path(deployment["prompt_path"]).read_text("utf8"),
            observation_top_k=plan["observation_top_k"], observation_token_limit=plan["observation_token_limit"],
            prefill_think=True)
        old = read_json(source/"binding.json")["behavior"]
        if fingerprint(tok.chat_template) != old["chat_template_sha256"]:
            raise ValueError("tokenizer template identity changed")
        old_ids = tuple(first["token_ids"])
        renderer.make_parser(malformed_policy="raise", summary_mode=NativeTagParser.SUMMARY_END_RIGHT_INCLUSIVE_V1)(
            old_ids, "stop")
        first_text = decode(old_ids)
        unwrapped_eos = (old_ids[-1] == tok.eos_token_id and "</think>" in first_text
                         and "<summary>" not in first_text and "</summary>" not in first_text)
        if not action_closed(first_text) and not unwrapped_eos:
            raise ValueError("saved first action has no supported stop")
        conditioned = tuple(prefix) + renderer.user_turn_suffix(body, old_ids)
        device = torch.device(deployment["actor_device"])
        actor, info = AutoModelForCausalLM.from_pretrained(deployment["actor_model_path"], torch_dtype=torch.float32,
                                                           output_loading_info=True, **local)
        if any(info.get(key) for key in ("missing_keys","mismatched_keys","error_msgs")):
            raise ValueError("incomplete actor snapshot")
        actor.to(device).eval()
        old_version, evidence = identify(actor, old)
        if old_version != first["policy_version"]:
            raise ValueError("saved sampling weights or behavior differ")
        new_version = fingerprint({**evidence, "behavior": {**old, "turn_protocol": CHAT_TURN}})
        generator = HFNativeGenerator(model=actor, tokenizer=tok, device=device,
            policy_version_getter=lambda: new_version, max_new_tokens=plan["max_new_tokens_per_step"],
            max_context_tokens=plan["max_context_tokens"], prefilled_think=True,
            precision_context=lambda: torch.autocast("cuda", dtype=torch.bfloat16,
                                                      enabled=deployment["dtype"] == "bfloat16"),
            structured_stop=True, summary_mode=NativeTagParser.SUMMARY_END_RIGHT_INCLUSIVE_V1)
        with operation("fixed_second_turn", seed=second["seed"], source_request=second["request_id"]):
            generated = generator.generate(conditioned, seed=second["seed"])
            if generated.finish_reason != "stop":
                raise ValueError("incomplete second action")
        atomic_json(root/"result.json", dict(first_round="archived", second_round="generated_once",
                    seed=second["seed"], prefix_sha256=fingerprint(conditioned),
                    generated_sha256=fingerprint(generated.token_ids), length=len(generated.token_ids)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run")
    parser.add_argument("new_output")
    args = parser.parse_args()
    replay(require_cluster_storage(args.source_run), require_cluster_storage(args.new_output))

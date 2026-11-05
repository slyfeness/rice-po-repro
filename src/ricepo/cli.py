"""Explicit offline entrypoints. No model initialization at import."""

import argparse
import json
from pathlib import Path
from .config import fidelity_issues, load_config, require_enabled, require_execution


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ricepo")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="read config and list unresolved choices; no model loading")
    inspect.add_argument("--config", required=True)
    train = sub.add_parser("train", help="explicitly enabled offline reference training inside Slurm")
    train.add_argument("--config", required=True)
    train.add_argument("--deployment", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate the fixed development-selected checkpoint on a held-out manifest")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--deployment", required=True)
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--test-manifest", required=True)
    evaluate.add_argument("--output-root", required=True)
    encode = sub.add_parser("encode-corpus", help="offline Diver encoding inside Slurm")
    encode.add_argument("--config", required=True)
    encode.add_argument("--deployment", required=True)
    encode.add_argument("--domain", required=True)
    encode.add_argument("--batch-size", required=True, type=int)
    summarize = sub.add_parser("summarize", help="aggregate existing evaluation JSONL inside Slurm")
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--domains", nargs="+", required=True)
    summarize.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        config = load_config(args.config)
        print(json.dumps({"execution": config.get("execution"), "issues": fidelity_issues(config)},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command in {"train", "encode-corpus", "evaluate"}:
        config = load_config(args.config)
        with Path(args.deployment).open(encoding="utf-8") as handle:
            deployment = json.load(handle)
        if args.command == "train":
            require_execution(config)
            from .hf_runtime import run
            result = run(config, deployment)
        elif args.command == "evaluate":
            require_enabled(config)
            from .holdout import evaluate_checkpoint
            result = evaluate_checkpoint(config, deployment, checkpoint_dir=args.checkpoint_dir,
                                         test_manifest=args.test_manifest, output_root=args.output_root)
        else:
            from .preparation import encode_corpus
            result = encode_corpus(config, deployment, domain=args.domain, batch_size=args.batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from .artifacts import atomic_json, require_cluster_storage, require_slurm
    from .data import read_jsonl
    from .evaluation import summarize as aggregate
    require_slurm()
    source, target = require_cluster_storage(args.input), require_cluster_storage(args.output)
    atomic_json(target, aggregate(read_jsonl(source), args.domains))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

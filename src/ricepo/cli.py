"""Explicit offline entrypoints. No model initialization at import."""

import argparse
import json
from pathlib import Path
from .config import load_config, require_enabled, require_execution, resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ricepo")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="read config and list unresolved choices; no model loading")
    inspect.add_argument("--config", required=True)
    expand = sub.add_parser("expand", help="print resolved recipe and research provenance without runtime I/O")
    expand.add_argument("--config", required=True)
    expand.add_argument("--for-command", choices=("train","baseline","evaluate","encode-corpus","compare-retrieval","inspect"), default="inspect")
    baseline = sub.add_parser("baseline", help="evaluate the initial retriever or prompting policy")
    baseline.add_argument("--config",required=True)
    baseline.add_argument("--deployment",required=True)
    baseline.add_argument("--output-root",required=True)
    baseline.add_argument("--split",choices=("development","test"),default="test")
    check = sub.add_parser("preflight",help="explicit future Slurm environment inventory")
    for key in ("config","deployment","output"):
        check.add_argument("--"+key,required=True)
    train = sub.add_parser("train", help="explicitly enabled offline reference training inside Slurm")
    train.add_argument("--config", required=True)
    train.add_argument("--deployment", required=True)
    train.add_argument("--resume",action="store_true",help="restore only the latest complete checkpoint")
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
    summarize.add_argument("--deployment",required=True)
    summarize.add_argument("--split",choices=["development","test"],required=True)
    summarize.add_argument("--partial",action="store_true")
    summarize.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command in {"expand","inspect"}:
        report = resolve(load_config(args.config),getattr(args,"for_command","inspect"))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    if args.command in {"train", "encode-corpus", "evaluate", "baseline", "preflight"}:
        config = load_config(args.config)
        with Path(args.deployment).open(encoding="utf-8") as handle:
            deployment = json.load(handle)
        if args.command == "train":
            require_execution(config)
            from .hf_runtime import run
            result = run(config, deployment,resume=args.resume)
        elif args.command == "preflight":
            from .preflight import preflight
            result = preflight(config,deployment,args.output)
        elif args.command == "baseline":
            from .holdout import evaluate_baseline
            result = evaluate_baseline(config,deployment,output_root=args.output_root,split=args.split)
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
    from .assessment import summarize_assessment
    if args.split == "development" and args.partial:
        raise ValueError("partial development aggregation is prohibited for checkpoint selection")
    require_slurm()
    source, target = require_cluster_storage(args.input), require_cluster_storage(args.output)
    from .dataset_bundle import verify_bundle
    with Path(args.deployment).open(encoding="utf8") as handle:
        frozen = verify_bundle(json.load(handle))
    rows = (json.loads(source.read_text(encoding="utf8")) if source.suffix == ".json" else read_jsonl(source))
    atomic_json(target, summarize_assessment(rows, frozen.identities[args.split], split=args.split,
                                             manifest_sha256=frozen.manifest_sha256[args.split],
                                             scope="partial_diagnostic" if args.partial else "complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

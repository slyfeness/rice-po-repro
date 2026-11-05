# Repository task constraints

Current phase: review publication of an already authorized bounded first trial. Read `docs/VALIDATION.md` and `docs/IMPLEMENTATION_LEDGER.md` before changing execution status. Publication itself does not authorize experiments or expand the earlier bounded trial.

- Keep source provenance and explicit implementation choices distinct from paper facts.
- Preserve native token IDs, full-vocabulary entropy, per-query anchor selection, fixed prefixes and fixed benchmark exclusions.
- Do not add automatic CI, model downloads, environment installation, or Slurm submission without a new request.
- Future cluster compute requires Slurm; large files, logs, reports, caches and checkpoints require `/file_storage01/home/<user>/...`.
- Existing experiments, environments and frozen configurations are outside this repository's scope.
- Four GPU resource allocation and four-rank program validation are separate acceptance steps.
- Do not publish credentials, local private data, model binaries or generated experiment artifacts.

## Human review delivery

- Publish each completed review stage on a separate branch and open a PR; never merge automatically or push main.
- Start the next review from the latest reviewed valid `validation/first-smoke` commit unless the user specifies a newer accepted baseline.
- Follow `docs/reviews/README.md`: disclose executed, failed, skipped and unrun work separately, and scan all proposed public material for sensitive information, size and licensing before upload.
- Public review evidence is limited to small sanitized summaries, manifests and SHA-256 references. Keep raw logs, account/host details, datasets, weights, trajectories and checkpoints private.
- Review publication does not authorize downloads, installation, new compute or the next training stage.

# Repository task constraints

Current phase: source implementation and publication only. Read `docs/VALIDATION.md` and `docs/IMPLEMENTATION_LEDGER.md` before changing execution status. No experiment has been authorized by publication of this repository.

- Keep source provenance and explicit implementation choices distinct from paper facts.
- Preserve native token IDs, full-vocabulary entropy, per-query anchor selection, fixed prefixes and fixed benchmark exclusions.
- Do not add automatic CI, model downloads, environment installation, or Slurm submission without a new request.
- Future cluster compute requires Slurm; large files, logs, reports, caches and checkpoints require `/file_storage01/home/<user>/...`.
- Existing experiments, environments and frozen configurations are outside this repository's scope.
- Four GPU resource allocation and four-rank program validation are separate acceptance steps.
- Do not publish credentials, local private data, model binaries or generated experiment artifacts.

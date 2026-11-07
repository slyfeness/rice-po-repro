# Repository task constraints

## Current delivery policy — 2026-11-07

- Deliver this RICE-PO project through the existing GitHub review branch and PR only; do not merge automatically.
- Do not generate ZIP, TAR, TGZ, 7z or other archives for delivery, backup or temporary source transfer. Use Git at a fixed commit or SFTP of expanded source files instead.
- The user requested removal of previously generated project delivery/source archives. Preserve expanded immutable execution source, model/data assets, checkpoints, native trajectories, logs and audit receipts.
- Historical reports and hashes describe former deliveries; they do not establish that those archives still exist. Do not rebuild them to repair a historical link.
- No cleanup or delivery action authorizes new model execution or training.

Current phase: source implementation and publication only. Read `docs/VALIDATION.md` and `docs/IMPLEMENTATION_LEDGER.md` before changing execution status. No experiment has been authorized by publication of this repository.

- Keep source provenance and explicit implementation choices distinct from paper facts.
- Preserve native token IDs, full-vocabulary entropy, per-query anchor selection, fixed prefixes and fixed benchmark exclusions.
- Do not add automatic CI, model downloads, environment installation, or Slurm submission without a new request.
- Future cluster compute requires Slurm; large files, logs, reports, caches and checkpoints require `/file_storage01/home/<user>/...`.
- Existing experiments, environments and frozen configurations are outside this repository's scope.
- Four GPU resource allocation and four-rank program validation are separate acceptance steps.
- Do not publish credentials, local private data, model binaries or generated experiment artifacts.

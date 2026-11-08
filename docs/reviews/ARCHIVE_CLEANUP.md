# GitHub-only delivery — 2026-11-07

User requested no more project archives and deletion of previously generated packages. Delivery now uses the existing review branch/PR; no automatic merge or new training is implied.

- Local RICE project/control directories:54 source/delivery archives deleted,129,048,974 bytes. Their project directories and the matching top-level RICE source-transfer files have no remaining archive files.
- Cluster project:70 regular source/delivery archives deleted,185,763,051 bytes. Inventory job175661 and deletion job175668 completed successfully. Every target was checked against the fixed inventory, resolved inside the project, matched inode/size/modification time, and was verified absent after deletion. The inventory contained only `control`, `incoming`, `releases`, and `exports` archive files; no asset-archive exception was required.
- Expanded execution source, models, datasets, vector assets, checkpoints, native trajectories, logs and historical audit receipts remain. Only archive files were unlinked; no directory was recursively removed.
- Historical archive paths, sizes and SHA-256 records describe past deliveries. They remain for provenance but are no longer download promises. Do not regenerate missing packages to satisfy historical reports.
- Future source deployment uses a fixed Git checkout or expanded-file SFTP. Small sanitized source/config/report changes go to GitHub. Private deletion manifests and path details remain private.

These were filesystem cleanup operations, not model tests or experiment validation. No GPU was requested and no optimizer or evaluation was run.

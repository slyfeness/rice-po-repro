# Continuation and retrieval diagnosis

Baseline: `768f60fbf9b16e22b643b720a6849bc6a7cec2a2`; existing PR #2.

1. Verify immutable update-one parents, add tested explicit child-run identity and global sampling schedule. New output directories; reference remains the original model. Run serial sampling for both continuations.
2. Analyze the existing 96 development trajectories and original qrels. Separate measured retrieval scores, terminal utility and matched-set coverage. No generation or development-driven recipe selection.
3. Perform global updates two and three for each method, save/reload checkpoints and evaluate all 32 queries at update three. Stop on unavailable RICE credit or substantive safety/identity failures.
4. Integrate and independently benchmark batch generation on frozen long native histories, retaining the serial training comparison. Do not enable a failed acceleration gate.
5. Publish safe source, aggregate results and evidence hashes to existing PR; source ZIP stays in private bulk experiment storage.

Budget at entry: 10.791666444444445 / 32 GPU hours; remaining 21.208333555555555. Every allocation counts.

Status: execution in progress; no new experiment results claimed.

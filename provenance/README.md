# Provenance

**Reference only; not runtime engine.** [sources.json](sources.json) is the source-of-origin record for this paper-based reimplementation.

- `primary_paper` identifies the RICE-PO paper and version.
- `unverified_author_code` means no RICE-PO author implementation was verified in the bounded public-source search on 2026-11-04. It does not assert that a private or unindexed repository cannot exist.
- `official_dependency` identifies public upstream dependencies such as Diver, ReasonIR, BRIGHT, and model repositories. It does not label them as the RICE-PO authors' code.
- Model and dataset revisions were inherited from the research freeze. Their payloads were not downloaded or revalidated in this task.
- The eight small [upstream reference files](../third_party/reference/README.md) preserve the original bytes and licenses. Their local Git blob hashes match the fetched upstream blob SHAs.

Every experimental interpretation should carry an evidence category: `paper_explicit`, `dependency_explicit`, or `reimplementation_choice`. A runnable implementation or future test pass does not establish that an undocumented choice matches the authors' setup.

This directory contains no personal contact details, cluster access settings, credentials, private records, experiment data, or model weights. No third-party source was imported or executed when creating it.

The review base and revision provenance are recorded in [revision.json](revision.json). The pinned upstream blobs and original license files are preserved.

# Data audit

Scope was intentionally limited to the already prepared Biology diagnostic
bundle. The private bundle receipt binds distinct training, development, and
test manifests; its recorded SHA-256 is
`5353bcedb9e2de30633430f32fe89671f4b7ad43031022be1262dace79296daa`.

The runtime accepted the selected training member and emitted an initial
retrieval evidence receipt before the first generation request. No test labels
were supplied to training and no formal test evaluation was run.

The actual queries, documents, qrels, manifest rows, document IDs, and storage
locations are not published here because they are data assets rather than
review-safe source evidence.

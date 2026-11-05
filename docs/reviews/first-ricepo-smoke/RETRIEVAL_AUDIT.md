# Retrieval audit

The stage used the same frozen Diver encoder and fixed Biology vector package
for the initial retrieval request. The request completed and an internal
initial-ranking receipt was written before generation began.

This stage did **not** execute a complete retrieval evaluation, score/ranking
comparison against an upstream reference, or a complete NDCG calculation.
Therefore it makes no retrieval-quality or upstream-asset-verification claim.
The local consistency and upstream-provenance conclusions remain distinct as
documented by the repository's asset receipt contract.

# Environment record

Execution used the pre-existing isolated project overlay and a locally frozen
actor. No packages, models, or datasets were downloaded for this stage.

| Component | Observed value | Status |
| --- | --- | --- |
| Python | 3.11 | executed |
| PyTorch | 2.10 with CUDA support | executed |
| Transformers | 5.5.1 | executed |
| Java | Temurin 21.0.12.1 | executed for the retrieval dependency |
| Retrieval overlay | project-local, read-only base environment plus project overlay | executed |
| Actor | frozen local 1.5B snapshot | identity-bound privately |
| Retriever | frozen local Diver snapshot | identity-bound privately |

Compatibility facts established by actual probes:

- The installed `generate` signature has no literal `use_model_defaults`
  parameter; passing it is rejected as an unused model keyword.
- A fresh explicit `GenerationConfig` with `do_sample=true`, temperature 1,
  top-p 1, and top-k 0 successfully generated a token from the frozen actor
  without that unsupported keyword.
- Tokenizing chat templates defaults to a dictionary-style container in this
  Transformers release. The runtime now explicitly requests the native token
  ID sequence.

The deprecation warning for `torch_dtype` was observed. It did not cause the
stage failure and has not been changed in this review branch.

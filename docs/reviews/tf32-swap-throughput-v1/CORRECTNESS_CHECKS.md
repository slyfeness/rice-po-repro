# Correctness and execution record

Scientific base: `bcc462feb1b1f29376a93d935199a9aaa4163b49`.
Expanded execution checkouts: `6483d7ed7efef88a0b71ff7eb07f1383588677d7`
plus the per-file benchmark overlays in EVIDENCE_INDEX.json. That intervening
commit only established the no-archive delivery policy. Existing source,
checkpoints and native experiment evidence were preserved.

## Actually executed

- CPU, local Python3.12.10: `PYTHONPATH=src python -m unittest discover -s tests -p test_inference_timing.py -v` — **5 passed**, zero failures/skips. These cover clock domains, unknown timing, source-labelled counter ambiguity, explicit co-resident memory permission and sticky identity-preserving partitioning.
- CPU, same interpreter: `PYTHONPATH=src python -m unittest discover -s tests -p test_inference_benchmark.py -v` — **3 passed**, zero failures/skips. The original manifest API remains intact. These are eight distinct test cases, not counts of repeated invocations.
- Static AST (12 changed experiment/helper/export/render files) and `git diff --check` were separate checks, not GPU tests. No unrelated formula or gradient suite was rerun.
- Native screen: S0/S1/S2 completed every preset request; all returned token/LP lengths aligned and LPs were finite. The fixed screen cap512 includes natural EOS, stop tags and length/format failures. Request completion does not mean a legal complete RICE episode.
- Selected S2 shared4096 action check: **16 requests, 15 valid actions, 1 invalid retained** ({'tag_structure': 1}); 7285 actual model tokens in 18.887s. Budget/mask alignment and environment LP=None checks: True/True/True. The invalid action was not repaired, discarded or resampled. This does not stress all144 requests to4096 output tokens or certify a full32768-token context.
- Co-resident validation175686 and single/dual saved-input workloads175694 retained **all48 ranking-ID sequences and NDCG values per workload**. No new query encoder, fusion, qrels, exclusions, reward or tie rule was introduced. Every floating score component was not independently re-regressed in this phase.
- Real closed loop: **52.359s**, 2 fixed training identities, 54 actual generation-stage requests, 11024 newly generated tokens, 171812 processed history/input tokens, 28 actual retrieval calls. Reused history is not counted as newly generated output.
- One original completed five retrieval rounds. The other failed at its first action: actual D0/last-valid NDCG1.0 was retained, terminal task utility0 was recorded separately, and no fictitious retrieval was submitted. Failure is not natural completion.
- The real branch group retained **5 new branches + original once**, six distinct branch IDs, five legal anchor summaries, and the same frozen native prefix. All native action masks/shared budgets/environment LP fields aligned. This scheduling probe uses the first actual legal Summary; `Anchor.entropy=0` is an unused schema placeholder, **not a measured entropy**. No entropy selection, variance/credit, advantage or PPO computation occurred.
- Version switch: both original policy payloads agreed (1 identity); both new payloads agreed (1 identity); old and new differ (True). Each replica actually constructed a new LLM from the existing updated checkpoint, with new process/IPC/cache lifetimes. Relative payload-file SHA-256 maps, including weights/tokenizer, are retained in the public receipt; no model bytes are published. This was not a version-string-only assignment. Existing real weight-change/independent-restore evidence is [reused](../precision-pilot-v1/DECISION.md).
- Four-GPU pure inference175696: four actual **NVIDIA A800-SXM4-80GB** workers, identical local payload identity, distinct Slurm device slots, TP1 each, all1152 preset requests completed per cache state without omission/duplication. Individual loads, token counts, memory and rates are in FOUR_WORKERS.csv. Aggregate throughput divides actual total tokens by the coordinator's common completion time.

## Failures, uncertainty and scope

- **FAILED**175680: same-process/JVM co-residency produced a native segfault diagnostic, Slurm exit11:0. The completed retrieval baseline and logs were kept. Unreturned in-flight generation cannot be reconstructed and is not claimed recovered. No OOM cause is established. Exactly one bounded process-isolation/cache-binding retry was performed and passed.
- Native NCCL teardown warnings were observed, followed by engine shutdown and terminal Slurm completion. They were not suppressed or represented as four-rank training validation.
- **NOT_MEASURED:** exclusive GPU-kernel prefill/decode percentages, hardware-level overlap, separate Diver activation-only peak, absolute unsampled device-memory peak, and per-kernel use of TF32. Native per-request intervals overlap and are not summed into exclusive device utilization. Polling uses a nominal0.2s wait plus collection overhead; memory/gauge peaks are sampled lower bounds.
- Three historical short screens did not all preserve live device names: S1 was checked live, selected S2 full-pool hardware/driver was checked live, and all dual/four workers record actual names. Other original screen records retain UUID/memory only; no historical node inspection was bypassed after access was denied. This limits exact hardware matching of those older screening points.
- Source-labelled `prompt_tokens_by_source` was ambiguous in the original metric-name reduction. It is explicitly `NOT_MEASURED` in the sanitized projection and protected by a focused CPU regression. Throughput uses raw output-ID counts and wall time; cache fractions use native per-request cached-token counts/input lengths, not that ambiguous counter. The published module also retains the legacy manifest functions; neither diagnostic-only difference changes executed sampling mathematics.
- Local asset equality is distinct from upstream provenance verification. `upstream_verification=not_verified` in the local-identity helper is preserved; this phase does not mint a new upstream provenance certificate.
- **NOT_RUN:** PPO/optimizer, entropy/credit, gradient/calibration repetition, development evaluation, BF16,32/128-query training, FSDP/four-rank training and full32768-token stress. Existing TF32 restricted-pilot qualification is reused; exact TF32/IEEE backend equivalence and new multi-engine Trainer integration are not certified.

## Sanitized actual command mapping

Private bindings are denoted `$PYTHON`, `$CONTROL`, `$SOURCE`, `$LOG_ROOT`;
only their paths are redacted. No account, node/address or credentials appear.
GPU submissions used the existing environment and offline assets, not installs.

| Jobs | Actual driver arguments | Resources / limit |
|---|---|---|
|175674|`$PYTHON $CONTROL/250-throughput-manifest.py`|CPU2,8G, no GPU|
|175677/175678/175679|`$PYTHON $CONTROL/251-throughput-screen.py S0/S1/S2 screen`|1GPU,6CPU,90G,30min|
|175680|`$PYTHON $CONTROL/252-coexist-probe.py` (failed historical path)|1GPU,6CPU,90G,30min|
|175686|`$PYTHON $CONTROL/253-isolated-coexist.py`|1GPU,6CPU,90G,30min|
|175687/175705|`$PYTHON $CONTROL/251-throughput-screen.py S0/S2 confirm`|1GPU,6CPU,90G,30min|
|175688/175700|`$PYTHON $CONTROL/251-throughput-screen.py S0/S2 screen`|1GPU,6CPU,90G,30min|
|175694|`$PYTHON $CONTROL/255-dual-scheduling.py`|2GPU,16CPU,192G,45min|
|175696|`$PYTHON $CONTROL/256-four-inference.py`|4GPU,24CPU,384G,30min|
|175698|`$PYTHON $CONTROL/257-throughput-export.py`|CPU2,8G,10min, no GPU|

The public `scripts/benchmarks/` counterparts replace private path constants
with explicit environment bindings. Individual private script/source hashes
are in EVIDENCE_INDEX.json. Hardware-only `srun --jobid=175705 --overlap
--ntasks=1 --cpus-per-task=1 --gres=gpu:1 /bin/bash
$CONTROL/259-own-device-metadata.sh` queried only its allocated device; the
same type of live check was used in175678. No extra allocation was created.

SLURM_JOBS.csv contains terminal states and UTC accounting times. Submit-to-start
dependency waiting is separate from execution: the four-card job's long pending
interval waited for the single-card validation; it was not a running service.
Actual allocated GPU time for this phase, including failed jobs, initialization,
repeats and persistence, was **2.402500 GPU-hours**. The export job itself was
queried after completion, rather than reported RUNNING from its own snapshot.

## Publication

Only source, small configurations, tests and aggregate/de-identified evidence
are included. Raw native histories, outputs, rankings, models, checkpoints,
cache and logs remain private. Existing licenses are unchanged; no third-party
weights/data/code bundle or new license is supplied. No automatic CI/merge,
training restart, archive or dependency installation was added.

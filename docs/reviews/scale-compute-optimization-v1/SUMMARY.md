# Scale compute optimization v1 — in progress

Original training jobs 176768/176798 remain cancelled. GRPO has complete checkpoint2; RICE has P1 actor snapshot only and must not resume as if Adam/RNG step1 were persisted. No original training restart is authorized.

Independent candidates A (fused full-original risk before step), B (native Summary predecessor full-vocabulary projection), C1 (existing action projection with exact CPU saved tensors). Default runtime unchanged. Existing source/tests reused; no new sampling/retrieval/reward computation.

Predeclared gates: B entropy max absolute error <=1e-5 and identical full candidate order/Top4 per query, native lengths unchanged. A/C log-prob max <=1e-4, conditional KL max <=1e-6, scaled loss <=1e-5, global gradient relative L2 <=0.001 and cosine >=0.99999; all gradients finite. Existing behavior-risk thresholds are not relaxed. Meaningful timing target >=1.10x paired median. C2 requires measured 8 GiB headroom; no unmeasured length threshold will be invented.

First CPU tests, then saved short/median/long GPU pairing; B eight original trajectories before fixed32. Results pending, no adoption claim. Online default and paused lineage remain unchanged. All heavy work via Slurm, private native records retained privately, no archives.

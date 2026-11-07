## 2026-11-07 TF32 GEMM training validation

Added isolated G_A/G_B four-query contract, frozen three-batch budget, exactly-one-update checkpoint handoffs and receipt-verified same-run resume. Legacy pilot budgets/defaults remain. Branch audit adds an optional acceptance callback while retaining its original default; the new path uses registered risk gates and existing stratification/expansion. Added explicit highest GEMM control and blocking engine-generation timing. No credit/objective/reward/parser/seed formula changes. [Measured speed, numerical trends, restore and limitations](reviews/tf32-gemm-training-validation-v1/SUMMARY.md); no32/128 launch or global validation upgrade.

## 2026-11-07 TF32 GEMM diagnostic

Opt-in `tf32_gemm_pilot` sets worker-only matmul `high`; `GemmInspection` verifies actual worker precision and installed Linear dispatch. Optional profiler/worker extension arguments retain old defaults. The independent B gate requires the frozen speed/IEEE-teacher evidence and exactly one two-query update. The B observer records the actual total PPO gradient/parameter change without repeating reference backwards; the policy-only gradient remains unmeasured. HF FP32 handling adds B without changing T/A behavior. No reward/credit/objective or partitioner changes. These are diagnostic capabilities and a new bounded numerical implementation choice, not a backend-equivalence claim. [Actual tests, results and limitations](reviews/tf32-gemm-diagnostic-v1/CORRECTNESS_CHECKS.md).

## 2026-11-07 TF32 throughput stage

Added optional native latency/cache/memory observations, deterministic sticky request partitioning and isolated immutable TP1 workers in `inference_*`; `VLLMEngine` metrics and low-memory co-residency require explicit opt-in. Default runtime and mathematical training modules remain unchanged. The benchmark Retriever loader explicitly overrides historical cache bindings. A labelled-counter ambiguity is marked unknown, and the original manifest API is retained. These are engineering capabilities/diagnostic fixes and optional deployment choices, not a change to credit, sampling temperature or PPO. [Actual measurements, failures and limits](reviews/tf32-swap-throughput-v1/CORRECTNESS_CHECKS.md).

## 2026-11-06 retrieval/vLLM stage

- Retained: original prompts/strict parser, native token histories/masks, fixed initial Reference, per-query entropy/advantages, population variance, thresholds, replacement/PPO and all historical artifacts.
- Added: independent per-stage vLLM native sampler, immutable actual-weight snapshots, process/KV lifecycle, full original/branch sampled-LP gates, opt-in two-device deployment template and explicit rejection of unsupported standalone vLLM holdout.
- Fixed during review: failed worker startup cleanup, unique RPC attempt paths and raw returned completions saved before probability conversion. No token repair, action filtering or reward change.
- New implementation choices: named vLLM request/stage RNG streams and phase-separated CPU residency. Retrieval composition remains a reported offline candidate only.
- Actual checks and runtime outcomes: [tests](reviews/four-gpu-deployment-review/TEST_RESULTS.md), [vLLM results](reviews/four-gpu-deployment-review/VLLM_RESULTS.md). Global validation flags are not upgraded; four-rank training, Retriever/vLLM co-residency and paper-scale performance remain unverified.

---

## 2026-11-06 连续训练修订

新增能力及原有缺陷逐项列于[本轮修改表](reviews/continuous-three-update/CHANGES.md)：独立检查点谱系、全局采样日程、成组原生动作恢复、真实检索退化分析与可选HF批量采样。工程修复和失效动作约定分别标明；不把源码接入等同于批量在线训练或分布式验收。原始缺失Summary证据不补写，旧失败产物保留。

---

## 2026-11-06 两阶段受限验收

本轮新增原生两阶段、动作mask/真实LP、同原始GRPO梯度诊断、验证式检查点、HF批量候选。9/16可估计、4个非零局部替换已真实参与更新，0个推理门控传播；开发性能未达标。类别、函数和验证状态见[修改表](reviews/two-stage-native-v1/CHANGES.md)，旧阶段资产与失败记录保留。

---

# Revision ledger

审查基准为 `d4b490a96f8ceb31392ca55abf31325bedb7078c`。开始时本地 main、origin/main 与远端公布的 main 均指向该提交，工作区干净。所有修改发生在独立 worktree 的 `revision/auditable-reference` 分支。本次交付使用独立本地提交和源码 ZIP。

逐项表 [REVISION_MATRIX.csv](REVISION_MATRIX.csv) 按任务原编号列出基准依据、改动类别、文件和函数、算法影响、测试源码、静态状态与未决事项。已有保护项也列入表内。实现选择的细节见 [PAPER_TO_CODE.md](PAPER_TO_CODE.md)。

| 范围 | 基准已有 | 本次变化 |
|---|---|---|
| 算法 | 原生 token 和掩码、分组熵与优势、原锚前缀、总体矩、双门控 | 保留这些语义；从有效配置传阈值、选点数、深度；GRPO 在分支之前分流 |
| 数据/检索 | 文件 SHA、显式 ReasonIR 选择、固定排除和 NDCG | 实际编码资产和目录发布、外部固定三清单、带身份的精确评测成员检查 |
| 训练 | 因果评分、PPO 剪裁、冻结参考、一次整批更新 | 独立旧概率路径、显式更新计划、探针和实际权重指纹、分项诊断 |
| 状态 | 原有分阶段 JSON，开发评估包在更新回调中 | 逐条证据、解析前原始结果、锁、原子 checkpoint、独立开发重试、完整提交点恢复 |
| 评估/成本 | 开发选模、正式测试、域宏平均 | D0 与初始 prompting 真基线、单一评估流、按阶段统计请求、失败、重放和时间 |
| 后续验证 | 未运行 | 测试源码已编写，未执行；模型数值、挂载点原子性、内存、GPU 和分布式仍需实测 |

`configs/paper.json` 和第三方许可保留基准字节。ZIP 的 `DELIVERY/` 目录包含提交身份、独立补丁、成员清单和 SHA-256；ZIP 自身的 SHA-256 在交付时按实际文件计算。

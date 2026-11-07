## 2026-11-07 HF评分与位置投影实测

[本轮结果](reviews/hf-compute-optimization-v1/SUMMARY.md)：HF TF32实际Tensor Core GEMM导致KL/loss/梯度偏移，拒绝；IEEE位置投影六轨迹数值通过、1次离线更新及Actor全参数重载一致，但仅快7.04%，低于事前10%门槛。3项CPU测试实际通过，预期RED与未运行S3分别披露。无生成/检索/在线训练/32或128验收，不升级全局验证字段。

## 2026-11-07 TF32 GEMM四查询连续验收

[真实结果](reviews/tf32-gemm-training-validation-v1/SUMMARY.md)：A/B各一次四查询真实RICE更新，B从同一配置与完整检查点继续至三更新；全部新权重/Actor/Adam/RNG恢复通过。三项针对性CPU检查实际通过，预期RED、准备失败及0项发现分别披露。端到端仅缩短3.68%，不满足速度门槛，不放行32。后两次仅登记分层分支概率抽查，旧一致性诊断超限且P99升高，稳定性通过只限本轮风险窗口。无开发评估/32/128训练；全局验证布尔字段不升级。

## 2026-11-07 TF32 GEMM受限验证

[本轮结果](reviews/tf32-gemm-diagnostic-v1/SUMMARY.md)：实际kernel切换、固定1152请求A/B/C、独立16历史概率、一次两查询RICE更新、新权重采样及独立Actor/Adam/RNG恢复已执行。4项针对性CPU测试实际通过，13项Slurm作业终态。B只为受限候选；线上/更新后历史一致性诊断超限项完整披露，本轮风险门槛通过不等于后端等价。没有32/128训练或开发评估，默认配置与全局验证布尔字段不升级。AST/JSON/差异检查单独标记，未重复旧基础测试。[检查范围](reviews/tf32-gemm-diagnostic-v1/CORRECTNESS_CHECKS.md)。

## 2026-11-07 TF32 推理调度

[实测结果](reviews/tf32-swap-throughput-v1/SUMMARY.md)及[检查范围](reviews/tf32-swap-throughput-v1/CORRECTNESS_CHECKS.md)：8项针对性纯Python测试实际通过；单/双引擎、真实检索闭环、已有更新后快照切换及四个独立TP1纯推理实际执行。原生失败动作保留，首次同进程共存的原生崩溃也保留。没有新PPO、熵/信用计算或开发评估；不升级全局验证布尔字段，不代表多引擎Trainer、四rank或论文性能验收。

## 2026-11-06 完整vLLM受限验收

[真实结果](reviews/four-gpu-deployment-review/SUMMARY.md)、[执行/失败记录](reviews/four-gpu-deployment-review/TEST_RESULTS.md)：完整五轮原生rollout/反事实/4096预算、24完整历史概率、RICE与GRPO各两个真实更新、实际权重切换、完整32题开发及独立模型/优化器/RNG恢复均已执行。独立失败环境、新后端RNG、小查询预算均显式标记；不代表严格论文性能、最大上下文容量或四rank程序通过。全局验证布尔字段不升级，历史声明保留其原时间范围。

---

## 2026-11-06 连续训练验收

本阶段实际执行情况见[逐项测试与失败记录](reviews/continuous-three-update/TEST_RESULTS.md)。64项针对性CPU测试、真实长轨迹采样概率与实际权重切换核验已执行；静态AST/JSON检查另记。原生unmapped ID的恢复仅限已定义失效环境，vLLM在IEEE FP32下实际完成80条长条件历史概率核验，完整两阶段/权重同步/Trainer仍未验收。每一历史报告保留原来的运行范围，公开模板与全局验证布尔字段不因此升级。

---

## 2026-11-06 两阶段受限验收

本阶段真实CPU、GPU、完整在线信用更新与跨进程恢复已执行，详见[本轮结果](reviews/two-stage-native-v1/SUMMARY.md)和[测试记录](reviews/two-stage-native-v1/TEST_RESULTS.md)。这些结果限定独立两阶段/失败环境，非严格论文性能或四rank验证；历史各阶段的“未运行”声明不改写。

---

# Verification status

Latest bounded online evidence: [failure-zero results](reviews/first-ricepo-smoke/FAILURE_ZERO_RESULTS.md). GRPO completed three updates and one reward-driven weight change, with three checkpoints independently reloaded. Both initial and trained policies evaluated all 32 development queries and completed zero legal five-round episodes. RICE collected all 65 scheduled new branches but had zero estimable anchors out of 13 and no weight change; its interrupted checkpoint is invalid and unpublished. This is an independent failure-environment experiment, not paper-performance or distributed validation. The source-only statements below are historical revision-phase status.

The first authorized trial is recorded in [FIRST_TRIAL.md](FIRST_TRIAL.md). After a shared-filesystem publication fix, the CPU suite recorded 219 passed and 2 skipped. A separate real 1.5B model diagnostic completed with zero optimizer steps. The source-only account below describes the preceding revision phase; full interactive RICE-PO validation remains pending.

**本次仅做静态审查。测试源码已编写，未执行。** 未导入项目模块、运行测试、下载资产、安装依赖、构建索引、训练、评估或提交 Slurm。模板执行开关和运行验证字段为 false。

| 范围 | 交付证据 | 后续需授权验证 |
|---|---|---|
| 源码与配置 | Python AST、内部导入和可解析调用签名，JSON/TOML、继承关系、差异白空、保护文件哈希 | 解释器导入、依赖及分支实际行为 |
| 信用/训练 | 既有公式测试和新增跨组件、梯度、零信号、小批次 fixtures | CPU 数值与梯度，实际采样概率与旧概率一致性 |
| 检索/数据 | 真实本地资产身份、三清单收据、精确查询集合、逐层比较工具 | 独立上游期望值、真实 tokenizer、模型、JVM、BRIGHT 内容 |
| 失败和恢复 | 原始 completion、阶段收据、外层检查点哈希、重放身份及锁测试源码 | 故障注入、目标文件系统原子性和双作业互斥 |
| 资源/结果 | 分项静态字节估计接口、分阶段成本记录 | 实际峰值显存、耗时、双设备全流程和分布式等价 |
| 发布 | 独立本地提交，ZIP 内的文件清单、逐文件 SHA-256 和提交信息 | 实验结果需另行授权获得 |

未来验收先执行纯 Python 与 CPU 张量层，然后按显式本地资产逐级启用。统计时分别列出已运行成功、已运行失败、缺资产跳过和未执行。用独立已知分布或固定探针比较方法和梯度。剪裁比例、梯度或参数变化无需每批严格大于零。对真实模型评分前先核实行为保持 eval 模式、相同采样和正确因果位置；资源估计须由实际测量补足。测试入口、分级和限制见 [TEST_TIERS.md](TEST_TIERS.md)。每个要求的基准证据、修改和未验证项见 [REVISION_MATRIX.csv](REVISION_MATRIX.csv)。

静态清点覆盖 55 个第一方 Python 源文件（含测试源码）、20 个 JSON 文件和 1 个 TOML 文件。静态导入名和可直接解析的内部调用签名已核对，文档本地链接已核对。16 份研究配置从继承链取得禁执行标记。12 个受保护文件（含原始论文配置、上游参考文件、来源清单和许可）与基准 Git 对象一致。此清点不证明运行时调用、动态库或数值行为正确。

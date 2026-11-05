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

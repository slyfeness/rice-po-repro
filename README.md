# RICE-PO 独立复现

交付规则（2026-11-07更新）：之后仅同步GitHub审核分支和现有PR，不再生成交付、备份或临时源码传输压缩包。已生成的项目交付包与源码归档按用户要求清理；历史报告中的ZIP路径和SHA-256保留为历史记录，不代表文件目前仍可下载，也不会据此重新打包。已展开执行源码及私有实验原始产物保留。

最新[TF32 GEMM诊断](docs/reviews/tf32-gemm-diagnostic-v1/SUMMARY.md)：实际worker由默认SGEMM切至Tensor Core，固定1152条件冷完成366.44→146.09秒（2.51×）。B完成独立数值筛选、一次两查询RICE更新、新权重采样和独立Actor/Adam/RNG恢复，仅为可选受限候选；更新后旧概率一致性诊断仍有超限项，不代表后端等价或128查询放行。[配置](docs/reviews/tf32-gemm-diagnostic-v1/BEST_RUNTIME_CONFIG.json)继续禁执行，未自动替换默认或启动后续训练。

此前[TF32 推理调度实测](docs/reviews/tf32-swap-throughput-v1/SUMMARY.md)：独占卡选 S2（144/32768/0.90）；双引擎与 Diver 共存采用独立进程、S0 的0.80/0.45布局，同负载完成时间约快1.57倍。四卡纯推理1152请求冷/热实测1725/6784 token/s，热缓存收益不能当作在线训练速度。[可选配置](docs/reviews/tf32-swap-throughput-v1/BEST_RUNTIME_CONFIG.json)保持禁执行；本轮没有PPO或32/128查询训练，未自动替换训练配方。

此前[混合后端受限验收](docs/reviews/precision-pilot-v1/DECISION.md)：TF32 rollout＋IEEE FP32训练完成两题真实RICE更新，5/8锚点可估计、2个非零局部优势，实际权重变化、新权重采样及独立Actor/Adam/RNG恢复通过。结论仅为`QUALIFIED_FOR_PILOT`；BF16 autocast A按优先级未新增验收，旧一致性FAIL不改写，未启动32/128查询训练。

此前[显式BF16权重验收：FAIL](docs/reviews/bf16-explicit-weights/BF16_TRAINING_DECISION.md)：同一16条原生历史、2591个采样token上，显式BF16比FP32权重/BF16 autocast误差更大，三项冻结门限均未通过。该阶段按停止规则未执行后续PPO或测速，保留IEEE FP32主线。

面向祝融，目标为 DeepSeek-R1-Distill-Qwen-1.5B 和 BRIGHT。最初源码审查基准为 `d4b490a96f8ceb31392ca55abf31325bedb7078c`。[论文 v3](https://arxiv.org/html/2605.26352v3) · [逐项修改表](docs/REVISION_MATRIX.csv) · [验证状态](docs/VALIDATION.md)

此前结果：[并行优化最终决策](docs/reviews/parallel-optimization-v1/DECISION_REPORT.md) · [推荐配置](docs/reviews/parallel-optimization-v1/BEST_RUNTIME_CONFIG.json)。匹配BF16方案未通过行为概率门槛，继续采用IEEE FP32。数组检索/batch1的完整906请求计时约快4.34倍，批量Diver编码2/4/8因排名变化被拒绝。四查询旧/新RICE与GRPO均完成更新、检查点和新权重采样，原生轨迹与奖励相同；整轮耗时仅约±1%变化，尚未证明明显端到端提速。128查询候选具备已记录的正确性/容量准备，但本轮未启动128查询连续训练；所有公开模板继续禁执行。

此前[检索协议、四卡部署与完整vLLM验收](docs/reviews/four-gpu-deployment-review/SUMMARY.md)保留原实验谱系：IEEE FP32 vLLM已完成两阶段五轮rollout、真实反事实信用、两次PPO更新、更新后权重切换和独立检查点/优化器/RNG恢复。该阶段完整32题失败调整分数RICE0.10583、GRPO0.11130，没有RICE性能优越性证据，不能与新协议/新评估口径直接拼接。

此前[HF三更新](docs/reviews/continuous-three-update/SUMMARY.md)和[早期单更新](docs/reviews/two-stage-native-v1/SUMMARY.md)保留原范围，不与本次新谱系拼接。

## 配置和方法

| 配置 | 实际用途 |
|---|---|
| `configs/paper.json` | 原基准中的论文常量，保留原文件 |
| `configs/paper_reference.json` | 每次 128 个查询，每查询 4 条原轨迹，跨轨迹选择 4 个摘要锚点，每锚点 5 条新分支加原分支，完整 5 轮 |
| `configs/smoke.json` | 只把查询数改为 1，优化器整批轨迹数随之从 512 变为 4 |
| `configs/retrieval_only.json`、`prompting.json` | 固定检索器的 D0、初始策略的完整交互评估 |
| `configs/grpo.json` | 同查询最终奖励归一化，直接进入共享 PPO 更新 |
| `configs/failure_zero_ricepo.json`、`failure_zero_grpo.json` | 已授权受限失败环境：4 查询更新、保留全部非法终止轨迹；明确偏离论文，见最新报告 |
| `configs/two_stage_ricepo.json`、`two_stage_grpo.json` | 两阶段环境Summary控制与显式动作mask，4查询诊断预算；数学信用/PPO保持 |
| `configs/ablation.json`、`configs/variants/` | 五个总分支、门控规则、选锚、局部归一化、查询组合或 PPO 轮数的单因素变体 |

从源码 checkout 部署，使用 `src` 作为 Python 模块根目录，保留同一 checkout 中的 `configs` 和 `prompts`。静态配置入口 `python -m ricepo.cli expand --config configs/smoke.json --for-command train` 只读配置，输出实际值、来源、偏离项、待填字段、部署字段名与指纹。本次未执行该命令。模板中训练总预算与真实数据清单路径留空，执行开关及运行验证标记为 false。[配置与选择规则](docs/REFERENCE_RUNTIME.md)

## 主方法

1. 用原查询检索 D0，同一冻结策略生成 N 条原轨迹。工具与提示 token 保留在因果上下文中。
2. 根据完整条件词表 Shannon 熵，在一个查询全部轨迹和轮次间选锚。从配对推理前的原生 token 前缀独立续跑至相同终点。原分支计一次，重复文本或奖励仍是有效样本。
3. 比较局部 NDCG@10 与最后一次检索的 NDCG@10。使用总体方差；在两个阈值条件同时成立时，把局部摘要优势复制到配对推理。门控关闭时保留轨迹优势。
4. 只对原始轨迹的生成 token 应用 clipped PPO 和给定采样历史的条件 KL。旧概率对应本批行为策略，参考模型固定在初始模型。实际权重身份、归一化、优化器小批次及更新诊断写入证据。[公式到代码](docs/PAPER_TO_CODE.md)

## 审计入口

- 数据准备保存三份划分、源行和正例关联的指纹。训练前读取绑定的评测身份，测试标签仅用于留出评估。完整结果要求精确覆盖全部 `(domain, query_id)`，按域内平均后计算域宏平均。[数据与复现边界](docs/REPRODUCIBILITY.md)
- 文档向量、行 ID 和收据作为一个版本目录发布。查询编码器必须与文档编码时的实际权重、分词器和编码配置匹配。[检索身份和上游逐层比较](docs/RETRIEVAL_AUDIT.md)
- 原始生成结果先于格式解析写盘。失败尝试、随机种子、前缀、成本和阶段收据保留。独占锁和已完成检查点约束恢复；开发集重试使用新尝试目录。[参考运行及恢复](docs/REFERENCE_RUNTIME.md)
- HF参考路径为单进程双设备，四卡以两个独立双卡作业并行使用。限定IEEE FP32路径的vLLM权重切换与32查询容量已有实据；四rank训练及128查询实际更新未执行。[祝融部署](docs/ZHURONG.md) · [测试分级](docs/TEST_TIERS.md)

论文报告的 4,011 条训练配额与实际样本 ID 是两种证据。复现明确记录了分支计数、查询拼接和训练预算的来源。代码来源与现有第三方许可见 [NOTICE](NOTICE.md) 与 [来源清单](provenance/sources.json)。

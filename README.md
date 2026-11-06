# RICE-PO 独立复现

面向祝融，目标为 DeepSeek-R1-Distill-Qwen-1.5B 和 BRIGHT。最初源码审查基准为 `d4b490a96f8ceb31392ca55abf31325bedb7078c`。[论文 v3](https://arxiv.org/html/2605.26352v3) · [逐项修改表](docs/REVISION_MATRIX.csv) · [验证状态](docs/VALIDATION.md)

最新受限结果：[连续三更新与检索退化报告](docs/reviews/continuous-three-update/SUMMARY.md)。RICE-PO与GRPO均完成三个真实更新及独立检查点恢复；RICE每次均有非零局部信用，推理传播仅一次且在末轮。完整32题失败调整任务分数：Initial 0.159289、GRPO3 0.055125、RICE3 0.070683；两者尚未恢复初始性能。两阶段/失败环境为独立实现，公开模板继续禁执行。

[VLLM IEEE FP32候选](docs/reviews/continuous-three-update/VLLM_STATUS.md)已通过80条长条件历史概率检查；同16请求/6734原生token生成155.78→23.30秒，约6.69倍。该候选尚未完成两阶段Trainer与训练后权重同步接入。真实HF整套采样加速约1.10倍。此前[第一更新报告](docs/reviews/two-stage-native-v1/SUMMARY.md)和[更早失败阶段](docs/reviews/first-ricepo-smoke/FAILURE_ZERO_RESULTS.md)保留各自历史范围。

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
- HF 参考路径为单进程双设备，已有受限训练及显存测量。四卡仅以两个独立双卡作业并行使用；四卡分布式、vLLM 权重同步及正式规模容量仍未验证。[祝融部署](docs/ZHURONG.md) · [测试分级](docs/TEST_TIERS.md)

论文报告的 4,011 条训练配额与实际样本 ID 是两种证据。复现明确记录了分支计数、查询拼接和训练预算的来源。代码来源与现有第三方许可见 [NOTICE](NOTICE.md) 与 [来源清单](provenance/sources.json)。

# RICE-PO 独立复现

面向祝融，目标为 DeepSeek-R1-Distill-Qwen-1.5B 和 BRIGHT。源码修订基于 `d4b490a96f8ceb31392ca55abf31325bedb7078c`。随后获授权的有限试跑完成了 CPU 测试和真实模型数值诊断；完整 RICE-PO 交互、训练与评测尚未验证。[论文 v3](https://arxiv.org/html/2605.26352v3) · [逐项修改表](docs/REVISION_MATRIX.csv) · [验证状态](docs/VALIDATION.md) · [首次试跑审核材料](docs/reviews/first-smoke/SUMMARY.md)

## 配置和方法

| 配置 | 实际用途 |
|---|---|
| `configs/paper.json` | 原基准中的论文常量，保留原文件 |
| `configs/paper_reference.json` | 每次 128 个查询，每查询 4 条原轨迹，跨轨迹选择 4 个摘要锚点，每锚点 5 条新分支加原分支，完整 5 轮 |
| `configs/smoke.json` | 只把查询数改为 1，优化器整批轨迹数随之从 512 变为 4 |
| `configs/retrieval_only.json`、`prompting.json` | 固定检索器的 D0、初始策略的完整交互评估 |
| `configs/grpo.json` | 同查询最终奖励归一化，直接进入共享 PPO 更新 |
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
- HF 参考路径为单进程双设备。分布式组件、vLLM 权重同步、数值一致性和峰值显存均需后续实测。[祝融部署](docs/ZHURONG.md) · [测试分级](docs/TEST_TIERS.md)

论文报告的 4,011 条训练配额与实际样本 ID 是两种证据。复现明确记录了分支计数、查询拼接和训练预算的来源。代码来源与现有第三方许可见 [NOTICE](NOTICE.md) 与 [来源清单](provenance/sources.json)。

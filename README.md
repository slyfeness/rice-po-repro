# RICE-PO reproduction

按论文 [RICE-PO: Turning Retrieval Interactions into Credit Signals for Reasoning Agents](https://arxiv.org/html/2605.26352v3) 实现的独立复现项目，目标平台为祝融，首个研究范围为 **DeepSeek-R1-Distill-Qwen-1.5B + BRIGHT**。

**交付状态：源码已实现，未运行。** 作者尚未公开可核实的 RICE-PO 代码。本仓库不是官方实现，也未复现出论文分数。本次没有导入或执行项目模块、运行测试、加载模型、处理数据、启动 Slurm、训练或评估；没有配置自动运行的 GitHub Actions。

## 已实现的主链路

```mermaid
flowchart LR
  Q[查询与初始检索 D0] --> R[同一旧策略生成 N 条五轮轨迹]
  R --> H[完整词表熵 / 同 query 选 top 4]
  H --> B[固定推理前历史 / 重采样分支至第 5 轮]
  B --> V[局部奖励方差与残差方差]
  V --> A[替换 summary 优势 / 门控传播至 think]
  A --> P[PPO clip + reference KL]
  P --> C[版本化 checkpoint 与开发集评估]
```

- 所有轨迹保留生成器原始 token IDs、半开区间 span、逐 token 生成掩码和策略版本；不通过 decode → encode 重建训练序列。
- 用完整词表 Shannon 熵选择 anchor，而非 sampled-token surprisal 或 top-k logprobs。top 4 在同一个 query 的全部轨迹、全部轮次间选择。
- 分支复用原始结果，从对应推理开始前的同一 prefix 重采样，并延续到最大深度 5。局部 NDCG@10 和最终 NDCG@10 均为 `[0,1]` 原始值。
- 使用总体方差，门控条件为 `influence_variance >= 0.05` 且 `residual_variance <= 0.03`。门控关闭时 think 保留轨迹优势；summary 优势采用替换，非叠加。
- 固定 Diver-4B + BM25；BRIGHT 固定排除文档与检索历史分离。参考实现提供精确稠密打分、Diver 上游 BM25 配方、显式混合归一化和 pytrec_eval 奖励。
- HF 参考入口连接采样、熵、分支、优势、更新和开发集评估。另有 vLLM 适配器及 DDP/FSDP 更新组件；它们均未经过运行验收。

## 目录与关键文件

| 路径 | 作用 |
|---|---|
| `src/ricepo/contracts.py` | 原始 token、span、轨迹、分支和信用记录 |
| `rollout.py`, `formats.py`, `backends.py` | 五轮交互、原生 token 解析、反事实分支、vLLM/熵后端 |
| `credit.py`, `objective.py` | 公式 (1)–(11) 的核心计算 |
| `modeling.py`, `training.py` | 因果 logprob/full-vocabulary KL、PPO 更新与 FSDP 组件 |
| `workflow.py`, `hf_runtime.py`, `cli.py` | 分阶段持久化与完整 HF 参考训练入口 |
| `retrieval.py`, `preparation.py` | 固定检索器与离线语料向量构建 |
| `data.py`, `manifest.py`, `evaluation.py`, `holdout.py` | 构建显式样本清单、检查泄漏、锁定开发集选出的 checkpoint，在独立 BRIGHT 测试集上计算域宏平均 |
| `configs/` | 论文常量、独立实现选择、部署模板 |
| `prompts/` | 论文 Figure 5 提示词转录与归属 |
| `slurm/` | 祝融作业模板，未提交 |
| `tests/` | 数学和边界测试源码，未执行 |
| `provenance/`, `third_party/reference/` | 来源版本、校验记录与保留许可证的上游参考文件 |

表中省略目录前缀的 Python 文件均位于 `src/ricepo/`。

## 论文确定的部分与实现选择

`configs/paper.json` 只固定论文明确给出的常量。`configs/implementation_choices.json` 提供有理由说明的参考约定，不冒充作者实验配置，例如 N=4、5 个新分支加原分支共 6 个结果、局部优势只中心化、完整词表 KL、summary-only 查询以及 PPO clip=0.2。

仍需未来实验填写：训练预算、独立训练/开发 manifest、实际本地资产与环境。论文给出训练集 4,011 条的域配额，但没有发布具体样本 ID，不能据此声称恢复了作者训练集。附录的“5 个分支”、正文查询组合方式和其他未报告参数的歧义，见 [论文到代码对照](docs/PAPER_TO_CODE.md) 与 [复现边界](docs/REPRODUCIBILITY.md)。

所有交付配置均为 `execution.enabled=false`。不存在已经通过的运行验收报告。后续获准实验时应复制配置、冻结独立 manifest 和全部实现选择，并另行完成 CPU/GPU 验收；本仓库没有自动下载、安装依赖或提交作业的步骤。

## 祝融部署

代码和轻量配置可位于控制目录；数据、模型、向量、缓存、日志、报告及 checkpoint 必须放在 `/file_storage01/home/<user>/...`。所有计算通过 Slurm。默认申请模板为 gre/professors、1 节点、1 task、4 GPU、24 CPU、384G、48 小时。

**四卡申请模板不等于四卡程序已验收。** 完整 HF 参考入口采用单进程、两个设备顺序执行，用于方法核对；四卡并行优化是后续独立验收事项。详见 [祝融部署约束](docs/ZHURONG.md)、[参考运行入口](docs/REFERENCE_RUNTIME.md) 和 [验证状态](docs/VALIDATION.md)。

第三方来源及许可见 [NOTICE](NOTICE.md)。

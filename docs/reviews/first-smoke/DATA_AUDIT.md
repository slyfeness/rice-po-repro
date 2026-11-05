# 数据状态与未运行检查

本阶段没有下载、构建或评测 BRIGHT / ReasonIR，没有可供报告的真实训练、开发、测试样本数量。GPU 探针使用合成短句，与 BRIGHT 查询无关。

来源声明继承自 `provenance/sources.json`，属于待准备的版本目标，不是本阶段实际内容验证：

| 资产 | 声明 revision | 内容与划分状态 |
|---|---|---|
| `xlangai/BRIGHT` | `3066d29c9651a576c8aba4832d249807b181ecae` | 未取得本项目绑定的三划分与正式查询集合证据 |
| `reasonir/reasonir-data` | `0275f825929b206d4ead23d34b4f8a50d4eddbc8` | 未准备或核验源行与正例映射 |

三清单 bundle、域计数、划分隔离、源正例映射和官方评测查询集合审计均为**未运行**，不能报告“零泄漏”或“作者样本身份一致”。4,011 是论文报告的训练配额，不是本项目实际训练数。

对应源码接口为 `src/ricepo/manifest.py`、`data.py`、`evaluation.py`、`holdout.py`；`tests/test_dataset_bundle.py`、`test_data.py`、`test_evaluation.py` 的合成 fixtures 包含在已执行的 CPU suite 中。它们验证代码行为，不能证明真实数据已经检查。部署模板 `configs/deployment.example.json` 的真实清单与 bundle 指纹继续留空。

下一阶段需先制定开发集来源、抽样与去重规则、正式测试重复文本的处理方法，再授权准备并冻结三划分。不得从正式测试题补开发域，不得静默丢弃无法映射正例。本 PR 不公开第三方原始样本或标签。

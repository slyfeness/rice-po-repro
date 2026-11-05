# 祝融部署约束与后续执行顺序

本次只交付源码与模板，没有部署到集群、申请资源或提交实验。

## 存储

建议未来创建独立项目，不覆盖现有实验目录、环境、历史结果或冻结配置。代码/轻量配置可放在 `/persist_data/home/<user>/projects/rice-po-repro`。其余一律放在 `/file_storage01/home/<user>/rice-po-repro/`：

```text
assets/       # 本地模型、BRIGHT/ReasonIR 原始数据
corpora/      # 领域语料与稠密向量、ID 顺序和 receipt
manifests/    # 独立样本清单、训练/开发/测试隔离记录
cache/        # Hugging Face、Torch、vLLM、Triton、JVM 等缓存
tmp/          # 临时文件
runs/         # 轨迹、熵、分支、优势、checkpoint、评估
logs/         # Slurm stdout/stderr 及其他日志
reports/      # 环境、静态/运行验收与结果报告
```

`%u` 是 Slurm 在输出路径中展开的用户名，不是 shell 环境变量。模板的日志路径为 `/file_storage01/home/%u/rice-po-repro/logs/`；提交前必须先创建目录，或通过 `sbatch --output=<绝对存储路径> --error=<绝对存储路径>` 指向已经存在的日志目录。不要让 Slurm 回退到代码目录中的默认 `slurm-*.out`。

## 资源与程序拓扑

`slurm/reference_train.sbatch` 针对 HF 参考程序申请 gre/professors、1 节点、1 task、gpu:2、24 CPU、384G、48 小时。四卡候选额度另列于 configs/zhurong.json。不约束节点或 GPU 型号，不逐张试卡。

参考训练器使用两个显式设备：actor/reference 在一个设备，固定检索器在另一个设备。它重在完整可核对的方法链路，尚未验证吞吐、显存和依赖兼容性；不会声称利用了四张卡。`training.wrap_fsdp` 与分布式 PPO 归一化是另行提供的组件，并未接入一个已验收的四进程生产训练器。获配四卡和程序支持四卡必须分别验收。

## 后续获得实验授权后

1. 只读核对真实项目 AGENTS.md、阶段任务卡、现有环境和可用资产。优先 SFTP 传输源码，复用环境，不自动安装依赖。
2. 冻结本地模型版本、领域语料、独立训练/开发 manifest 和选择规则。记录作者未公开的样本选择差异；不使用测试集选择 checkpoint。
3. 在 Slurm 作业内执行 CPU 测试、数据处理、向量构建、哈希和依赖验收。登录节点仅做轻量检查、编辑、传输和提交。
4. 单独授权最小 GPU 冒烟测试，核对生成 IDs、跨度、采样分布、分支 prefix、固定检索器、old/reference 概率和梯度。不得将配置中的 `runtime_validated` 直接改为 true 当作验收结果。
5. 通过上述验收后再确定预算和完整训练。大容量输出路径必须指向存储盘。`execution.enabled` 和提交脚本授权变量的改变必须对应实际新实验授权，本次不修改。

状态分别核对排队时间、batch 启动时间、环境/模型初始化时间和实际计算时间。使用 squeue、sacct 和作业日志，不能把 RUNNING 当作初始化完成。不默认创建定时监控，不绕开 Slurm 在登录或计算节点运行计算。

## 入口（仅供以后获准时使用）

`ricepo expand` 按命令静态展开配置和来源，`ricepo preflight` 供未来 Slurm 环境盘点，`ricepo baseline` 评估 D0 或初始策略，`ricepo inspect` 读取配置并列出未决参数；`python -m ricepo.manifest` 依据显式样本清单产生独立 train/dev/test 数据清单；`ricepo encode-corpus` 构建本地 Diver 语料向量；`ricepo train` 启动 HF 参考链路；`ricepo evaluate` 只评估开发集已选定的 checkpoint；`ricepo summarize` 汇总已有评估记录。计算入口要求 Slurm；准备和训练阶段分别需要相应配置和授权。此交付没有执行其中任何命令。

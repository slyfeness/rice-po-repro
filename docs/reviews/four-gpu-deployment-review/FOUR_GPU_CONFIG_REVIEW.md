# 四 GPU 部署审核：原文件、测量依据与候选布局

2026-11-06；基准 `7e358626`；现有 PR2。本报告供人工/ChatGPT 审核，**不是正式四卡训练放行记录**。本轮不再受原32 GPU小时停止上限约束，实际成本仍逐作业登记。

## 判断

可以沿用 SwAP 的独立 TP1 副本、批量调度、Prefix Cache、CUDA Graphs 思路；不能照搬其 BF16、FLASH_ATTN、T0.6/top-p0.95/top-k50 或四副本常驻布局。当前 RICE 的行为基准是 IEEE FP32、T1/p1/k0，完整训练还需要固定 Reference 与 Diver 检索器。

优先验收现有两卡布局中的**分阶段复用**：Actor 卡在 vLLM 生成与 HF 评分/更新之间切换；Retriever 留在第二卡。两个方法可各用一组双卡并行。它牺牲加载时间，先解决正确权重同步、缓存隔离和完整两阶段生成。是否改为推理/检索共卡或集中推理服务，等待本报告审核及实际显存验证。

## 原配置与可核对来源

来源别名 `$SWAP_ROOT` 对应用户既有 SwAP 项目，真实绝对路径只在私有来源清单保存。源文件：`configs/deployment-four-gpu-fast-generation-v2.json`、`slurm/14_four_gpu_config2.sbatch`；实测 `outputs/zhurong-four-gpu-generation-config2-169532/RESULTS.json` 和 `block-0/batch144.log`。本次读取 SwAP checkout HEAD=`d1854a0a14842645f33aa2527e229448a2641f33`，**上述配置/脚本为未跟踪文件，不能声称由该提交唯一绑定**；原件逐文件SHA见 `SOURCE_INVENTORY.json`，脱敏内容见 `SWAP_CONFIG.json` 和 `SWAP_SLURM.txt`。未修改 SwAP 文件、环境或任务。

| 原 SwAP 设置 | 文件/日志实值 |
|---|---|
| GPU / 布局 | 同一节点四张 A800-SXM4-80GB；四个独立实例 |
| Slurm | 1 node /1 task /4 GPU /24 CPU /384G主存 /48h |
| TP / PP / DP | 每实例 TP1、PP1、DP1；进程级四副本，不是 TP4 |
| 精度 / attention | BF16 / FLASH_ATTN；KV cache dtype=auto |
| 并发 / 显存比例 | 每卡144 /0.90 |
| 最大上下文 / batched tokens | 16384 /32768 |
| 单次生成预算 | 12000，训练T0.6、p0.95、k50 |
| Prefix cache / chunked prefill | 开启 / 开启 |
| Graph / compile / async | FULL_AND_PIECEWISE 实际捕获 /Inductor /显式开启 |
| 滑窗 | full attention，hf_overrides sliding_window=null |
| 通信 | 推理副本不需跨卡模型collective；同节点存在不等于已核验NVLink/带宽 |

历史推理作业169532四设置轮换、每设置四次：144/.90/32768 中位数6338.66 token/s/GPU，四副本算术合计25354.65；并非当前RICE吞吐。零OOM/preemption，16个worker正常结束，零optimizer step。单worker自身峰值72464–72470 MiB；一份日志给出KV池65.88 GiB、2466960 token容量、Graph池实测0.28 GiB。这些是该BF16负载/版本的值，不能迁移成RICE FP32容量。160并发更慢，144/.90/16384仅慢0.088%，无须为后者微小差异承诺收益。

原独立副本推理可分布到不同节点，各自本地设备映射；需要分配/汇集请求，不能假定共享localhost。四rank训练的跨节点NCCL、网卡/IB、NVLink及带宽没有本轮证据，均不标已验证。现有RICE两卡单进程要求其两张卡在同一节点；两组作业可以分处不同节点。

## RICE 当前资源：实测与估计分开

既有 RICE/GRPO 续训作业实际使用 A800-SXM4-80GB、驱动580.65.06、Torch CUDA12.8；每作业2GPU/16CPU/192G。现有隔离环境Py3.11.16、Torch2.10.0+cu128、Transformers5.5.1、vLLM0.19.1；本轮CPU175005重新读取两环境安装元数据，Triton3.6.0。安装元数据不能代替新GPU作业实际加载验收。

| 开销项 | 依据与范围 |
|---|---|
| RICE Actor设备峰值 | allocated 60611153408 B（56.45 GiB），reserved 82367741952 B（76.71 GiB）；包含Actor、Reference、Adam、梯度、激活等，**不是Actor权重单项** |
| GRPO Actor设备峰值 | allocated 55430890496 B（51.62 GiB），reserved 84322287616 B（78.53 GiB） |
| Diver设备峰值 | allocated 8150408704 B（7.59 GiB）；BM25在CPU，不能归为GPU模型 |
| 单FP32模型静态权重 | 实际checkpoint339张量、1777088000元素，7108352000 B（6.62 GiB）；按文件头计数，不是独立GPU分项实测 |
| Reference / gradients | 各约6.62 GiB静态量级；实际分项峰未隔离 |
| Adam一二阶状态 | 约13.24 GiB静态量级，另有小状态/分配开销 |
| vLLM FP32副本 | 已验收候选显存池比例0.80；早期worker NVML查询失败，不能将父进程0字节当峰值。新完整rollout175004已记录模型加载6.71GiB，父进程PyTorch峰不含worker；同步全设备峰仍未测量 |
| KV / Graph | 完整rollout175004的worker日志：KV54.59–54.63GiB、Graph池0.17GiB；这是池大小，不是全设备峰；SwAP65.88/.28 GiB不能套用 |
| 长历史采样 | 固定5869–6956 token前缀、最多512新token：HF16请求155.78s，IEEE vLLM23.30s，同6734原生输出token；约6.69倍仅生成段 |
| B64条件吞吐 | 28201 token /33.06s =853.12 token/s；不是4096动作预算的训练吞吐 |
| 初始化 / 同步 | 前一IEEE引擎初始化86.78s；新实现完整FP32快照+新进程重载，实际传输/初始化需单独计时 |
| 检查点 | 两方法已有实际保存、权重/优化器/RNG重载；与vLLM完整衔接仍单独验收，不能从已有HF成功推出 |

当前训练卡reserved接近80GiB，直接再放vLLM副本没有容量依据。即使按显存池0.8×80GiB再加Diver7.59GiB计算看似可容纳，仍未覆盖真实上下文/KV/Graph/非Torch开销；不能仅凭相加就批准共卡。尚无分离的 Prefill/Decode kernel 计时，已有总生成率不作两者测量值。

## 三种布局比较

| 布局 | 资源/公平性 | 同步和瓶颈 | 状态/复杂度 |
|---|---|---|---|
| A：两组独立双卡，Retriever卡兼推理 | 每组Actor+Ref+Adam一张，Diver+vLLM另一张；可公平并行且独立权重/目录。两组可跨节点，各组本地两卡 | 每次更新发布独立FP32快照并确认推理加载；推理与Diver会争用同一卡，pool需验证，不能直接沿用.8或.9 | 候选，**未验证共存**；中等改造 |
| B：训练与推理分离 | 一张训练卡按方法顺序更新、一张Diver服务、两张各自方法推理；单节点便于生命周期，跨节点需认证传输/独立端点 | 两推理卡无检索争用；训练卡成为串行瓶颈，不能同时持有两套训练状态；严格按冻结请求与预算比较 | 候选，服务化/双方法状态切换较复杂；未实跑 |
| C：两组双卡，Actor卡分阶段复用 | 每组GPU0生成时仅vLLM，评分/更新时仅HF；GPU1固定Diver。两个方法独立并行、原布局保持 | CPU offload模型和Adam，快照重载，结束旧进程销毁KV；同步和编译加载可能抵消短批收益 | **本轮先验收的保守实现**；适合完整正确性门控，未证明最快 |
| SwAP式四生成后四评分 | 对大规模独立响应适用；RICE每轮还需固定Retriever、熵选点与局部分支，不能四卡全占而省略这些步骤 | 若时间复用四卡，需要新的跨进程调度/模型阶段；训练仍不是已实现FSDP4 | 研究备选，不直接启动 |

建议先完成C完整验收，再用真实阶段计时决定A/B是否值得。若训练更新占比仍高，只优化vLLM不会同比加速整轮。

## 建议配置（待审核，未固化正式训练）

已有两卡诊断沿用每作业1节点/1进程/2GPU/16CPU/192G，最多两个作业同时获配。未来如批准单个四卡allocation，资源模板1节点/1task/4GPU/24CPU/384G/48h，再通过Slurm step隔离两组设备；不手写物理GPU编号、不指定节点/型号。

RICE候选：FP32、`TRITON_F32_DEFAULT=ieee`、TP1/PP1、max_model_len32768、max_num_seqs64、max_num_batched_tokens16384、memory_utilization0.8、Graph与Prefix Cache开启、TRITON_ATTN、full attention。仅在引擎独占其阶段GPU时使用0.8；共卡方案须另测。两阶段共享4096模型token，T1/p1/top-k disabled，环境token无直接损失，停止标签不改变采样logits。

默认Triton TF32候选此前3/80概率门限失败；IEEE修正后80/80通过。不得恢复TF32或改BF16追求SwAP数字。更长4096动作/五轮/实际PPO后策略刷新需本轮独立通过。

## 本轮完整 rollout 的实测补充

175004已完成真实两阶段/4096共享预算/五轮/四原轨迹/20新分支和全部24历史概率检查；4原轨迹完整、14分支完整、6分支按既定非法终态保留。实测33417新模型token/454.77s生成调用（包含内部生命周期加载）=73.48 token/s；采样/检索/熵/信用工作量522.83s，整allocation1204s，另包括初始checkpoint校验/加载和末尾概率评分。详细误差与信用见[VLLM_RESULTS.md](VLLM_RESULTS.md)。这不是此前64并发短条件853 token/s的同一计时口径。

当前Trainer每个查询批量原始4条、每个锚点批量新分支5条，远小于64并发容量。完整上下文采样通过不等于利用率已优化。两个在线更新作业中还观察到大体积checkpoint序列化、重复读取/校验和bulk staging复制的存储等待；新增GPU不能直接消除这一阶段。固定Actor/Reference/PPO及正确恢复前提下，下一步部署审核应同时考虑请求跨查询聚合与检查点/快照I/O，不能只选择最大推理并发。

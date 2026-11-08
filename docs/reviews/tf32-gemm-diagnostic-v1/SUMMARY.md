# TF32 GEMM diagnostic — bounded candidate qualified

基准 `5d4b4ff5287941abb365637b222239ae29ce9909`；现有 PR #2，未合并。**建议采用 B 作为可选的受限试跑配置，保留既有默认配置。** 本轮仅一次两查询 RICE 更新，不放行长期或128查询训练。

## E1 — 实际内核与同负载速度

**MEASURED：GEMM 开关有效。** 安装的 vLLM Worker 会用 `VLLM_FLOAT32_MATMUL_PRECISION` 重设 PyTorch 精度。A 的默认值是 `highest`；B 仅在推理 worker 设置 `high`，实际 RPC 确认 `allow_tf32=true`。原生 profiler 捕获的112个批量 Linear kernel 从 `ampere_sgemm` 切换为 CUTLASS `tensorop_s1688gemm`；单行 decode 的 GEMV 未改变。原始 trace 私有保存，[内核证据](KERNEL_COMPARISON.json)记录名称、数量及 SHA。微型探针的 kernel 累计时间不作为完整负载提速证据。

同一1,152原生条件、同一初始模型/种子/S2（144/32768/.90）、32768上下文、512输出筛选上限；每组先清空前缀缓存，再热回放。Think/Summary各576请求，原轨迹303、反事实前缀849。A复用历史175705；核对相同模型/manifest/环境与 A800-SXM4-80GB/驱动580.65.06，未重复已有效的基线。A/B均 FP32权重、Triton TF32 attention；C 为 BF16/FlashAttention速度参考。

|组别|冷缓存输出token|冷完成秒|冷token/s|热完成秒|热token/s|初始化秒|
|---|---:|---:|---:|---:|---:|---:|
|A 默认GEMM|225745|366.44|616.05|287.60|783.98|86.57|
|B TF32 GEMM|226505|146.09|1550.41|122.47|1844.87|79.75|
|C BF16参考|223494|84.22|2653.56|60.40|3705.79|92.93|

B冷缓存 makespan **2.508×，缩短60.13%**，达到预先要求的≥1.5×；输出量增加约0.34%，没有删除失败/EOS请求。包含初始化的冷池总完成时间453.01→225.85秒，约2.01×。未发生OOM或preemption；实测活跃并发达到144。A/B冷缓存命中67.69%/67.64%，未命中输入token分别1,489,556/1,491,860；显存采样峰值75,519/75,523MiB（非绝对峰值）。

**MEASURED native请求延迟均值：** Prefill 3.123→0.942秒，Decode35.708→13.851秒，Queue81.582→28.678秒，TTFT87.108→31.445秒。Prefill相关请求间隔明显下降；这些并发区间重叠，不能相加或冒充GPU独占阶段占比。完整负载的独占prefill/decode/queue百分比为 **UNKNOWN**。冷缓存为主结论；C热命中99.79%，明显高于A/B约77.6%，不能把热吞吐差全部归因精度。[完整CSV](GEMM_COMPARISON.csv)及[延迟/缓存/输出分布](BENCHMARK_DETAILS.json)。

## 独立数值与实际 PPO

冻结计划175841早于生成；排除237个历史校准/Pilot条件，固定16个新历史（Think/Summary各8，1057–11281输入token）与新种子，无结果筛选。使用保存历史的完整剩余4096共享动作预算，未重编码或重采。[冻结身份](VALIDATION_PLAN.json)及[逐请求结果](NUMERICAL_VALIDATION.json)。

**MEASURED：** 3,782真实采样token，绝对log误差mean0.002450、P990.015270、max0.075853；ratio范围0.96394–1.07880，越界0%，极端尾部0%，ESS比例0.999979。通过固定≤1%越界、≤0.05%极端尾部风险筛选；HF teacher实测IEEE FP32。

**MEASURED 在线175853：** 两固定训练查询×四原始轨迹全部保留，4/8合法五轮、4/8非法终止；8锚点各五新分支，共40个独立新分支（7个非法终止）全部保留。5/8锚点可估计、3个非零局部信用、5个Summary替换；3个不可估计锚点按既定语义保留轨迹优势。推理传播0次：5个可估计锚点的局部方差均未达到0.05，未降低门限；其中末轮残差为0按定义记录。原轨迹最终任务奖励第一题[0,0,0,1]、第二题[1,1,1,0]。局部奖励仍为真实检索NDCG，失败0仍仅为环境效用。

实际一次optimizer.step，8序列/10,036动作token，正/负/零优势7307/2525/204；policy loss−496.05495、初始reference KL0、总梯度范数1782.18823；FP32参数变化L2=0.03660403，1,546,270,516个元素变化，Reference无梯度。测的是实际总PPO梯度，未额外隔离policy-only梯度或重复旧梯度实验。40分支不进入优化批次。

所有原轨迹与分支评分记录共76,886位置（含分支复用历史，**不是新生成token数**），risk越界0%/极端0%；其中max误差0.18384使历史一致性诊断失败，原阈值保留。真实新生成量为原轨迹10,036、反事实37,625；探针306另列。实际D0两次、后续检索128次。[原始奖励/信用/更新聚合](ONLINE_EVIDENCE.json)与[成本计数](ONLINE_COST_AGGREGATES.json)，嵌套成本计时不得相加。

新权重实际重新序列化、原worker销毁、新worker加载并采样，KV不跨权重复用。更新后306token探针：mean误差0.01206、P990.15549、max0.19165，ratio越界1/306=0.327%、极端0%、ESS0.999229，**本轮风险门槛通过，历史/匹配一致性诊断未通过**。因此只认定受限候选，不认定后端等价或长期稳定。

Checkpoint发布、模型重加载及独立175862恢复Actor/Adam(step=1)/CPU+CUDA RNG均通过；恢复额外step=0，原产物未修改。[恢复证据](INDEPENDENT_RESTORE.json)。更新完整driver墙钟1138.86秒（19分钟），含初始化、全量概率复核、梯度/权重审计、检查点和新权重探针；不是纯训练吞吐，也无同查询A/B整轮提速结论。父HF进程torch峰值42.714/8.140GB与vLLM整卡采样峰值口径不同，不能相加或互相代替。未做开发评估。

## E2 — 旧训练独占耗时与估计

CPU作业读取原32查询与四查询日志，以UTC终点减elapsed重建区间，按叶子/宏阶段并集拆分。每个run所有独占行之和等于真实driver墙钟，嵌套generation/rollout/credit/update/evaluation没有重复相加。Sampler offload+快照+引擎启动的嵌套实测timer单列；未记录的初始化/IO等原因保持UNKNOWN，不把残差统称初始化。

旧32查询RICE总7614.27秒：训练generation host调用2357.55秒（30.96%，已扣生命周期242.44秒）、检索546.27、熵/信用534.26、HF评分/PPO2207.68、checkpoint214.54、评估718.55秒；其余身份校验/开销/未知残差见[独占CSV](TRAINING_COST_BREAKDOWN.csv)。host调用仍含传输、格式与证据IO，纯GPU生成占比为UNKNOWN。四查询RICE生成占18.76–19.18%，checkpoint约26.6%；GRPO32生成占15.07%，HF/PPO37.19%。

**ESTIMATED Amdahl：** 若只有上述训练生成组件快1.5倍，旧32-RICE墙钟最多按该口径缩短10.32%（1.115×）；已知评估生成也快1.5倍，则缩短12.57%（1.144×）。若训练生成真能保持本次2.508×，同口径约缩短18.62%；真实在线并发/前缀和输出分布不同，此数不是实测。纯GPU与host开销尚不能完全拆开，实际收益可能更小。[六个run估计与证据](AMDAHL_ESTIMATES.json)。

## E3与最终建议

[CPU缓存审查](CACHE_AFFINITY.md)：既有group从未拆散，但不同group共享前缀；四worker额外复制874,160前缀token，可解释约19个百分点理想冷命中损失，与旧单/四卡差的量级一致。没有经过验证的更优分配，未改调度器、未追加GPU对比。

B作为可选受限配置：FP32权重，推理worker `TRITON_F32_DEFAULT=tf32` + `VLLM_FLOAT32_MATMUL_PRECISION=high`；HF仍IEEE FP32；S2 144/32768/.90，TP1。详见[禁执行配置](BEST_RUNTIME_CONFIG.json)与[机器可读决策](FINAL_DECISION.json)。已验证默认值、全局validation字段不升级。

**ESTIMATED：双引擎Trainer不再优先于B的实际端到端验收。** 旧双引擎1.57×属于旧数值配置；它与B的叠加收益、Diver共存内存及Trainer权重生命周期仍UNKNOWN。本轮没有重复四卡或双引擎实验，也没有自动集成/替换正式配方。

13个本阶段Slurm作业全部COMPLETED/0，累计0.948611分配GPU小时（含初始化/保存/恢复，A历史基线不重复计费）；最多两卡并发，符合最多四卡且最少测试的范围。[全部作业](JOBS.csv)。4项新针对性CPU测试实际通过；AST/JSON/差异检查另列，不当作运行测试。版本/已安装上游源码hash见[环境](ENVIRONMENT.json)，未安装或修改环境。

仅更新现有PR，不合并；无压缩包。私有模型、数据、原生轨迹、checkpoint及原始日志继续私有保存，通过[证据SHA清单](EVIDENCE_INDEX.json)关联。完成后停止，不启动32/128查询训练。

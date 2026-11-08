# HF 计算优化：本轮结论 REJECT

HF TF32（S1）未通过预登记数值/梯度门限，不能采用。IEEE + action_positions_v1（S2）正确性通过，但 HF 评分与梯度累积仅加速 **1.0757×，耗时缩短 7.04%**，低于测量前固定的1.10×门槛。保持现有 IEEE 默认路径；S2作为明确 opt-in 的显存优化保留，不自动切换正式配方。

| 方案 | 真实范围与结果 | HF评分+梯度累积，秒（两次） | 峰值allocated / reserved，十进制GB |
|---|---|---:|---:|
| S0 IEEE完整Head | 六轨迹基线 |65.6285 /65.6184|61.070 /70.680|
| S1 TF32完整Head | 两轨迹数值失败，立即停止扩大 |未进行六轨迹测速|未扩展测量|
| S2 IEEE位置Head | 六轨迹正确性通过 |61.0086 /60.9974|54.150 /64.563|
| S3 TF32位置Head |未运行，S1先决条件失败|未运行|未运行|

S2降低峰值allocated约6.920GB（11.33%）；整卡NVML峰值72.044→65.914GB。NVML在模型加载后每0.2秒采样，初始化显存峰值未测。模型初始化S0=137.270秒、S2=10.027秒；文件系统缓存状态不同，**不将此差异归为优化收益**。

## 数值、目标及梯度

门限在CPU176665写入冻结计划，早于GPU176670；完整门限、计划指纹、来源绑定见[FINAL_DECISION.json](FINAL_DECISION.json)。TF32的动作LP mean≤.02/p99≤.10/max≤.50、KL mean≤1e-4/max≤1e-3、loss缩放误差≤1%、全局梯度cos≥.99/相对L2≤.10，显著张量cos≥.95/相对L2≤.20。S2的LP max≤1e-4、KL max≤1e-6、loss≤1e-5、梯度cos≥.99999/相对L2≤.001，显著张量cos≥.999/相对L2≤.01。显著张量阈值为S0张量梯度范数≥全局范数的1e-6；全部张量仍保留统计。

- **S1，两条保存的真实轨迹**：LP mean .015054、p99 .126270、max .413221；KL误差mean .00026949、max .0472612；loss误差1.3663%；梯度cos .969674、相对L2差25.6987%。最差显著张量相对L2差68.96%。梯度有限仍不能抵消上述失败。Profiler确认实际从ampere_sgemm变为CUTLASS tensorop_s1688gemm；不是只检查配置开关。
- **S2，全部六条轨迹**：每个动作LP、完整词表条件KL及loss与S0比较误差均为0；全参数梯度cos .9999999986、相对L2差 .0000529810（0.005298%），通过全部门限。完整逐张量结果见[GRADIENT_COMPARISON.json](GRADIENT_COMPARISON.json)，逐动作统计见[PRECISION_COMPARISON.csv](PRECISION_COMPARISON.csv)。有限样本上的前向一致不等于所有硬件/历史逐位等价。
- 保留完整原生IDs、mask、EOS及工具上下文；动作t仅选择hidden[t−1]进入**原始151936维**LM Head，Reference取同一位置。Transformer计算范围与既有S0 compact评分相同；既有评分不处理最后动作之后无因果贡献的尾部，原始存档不变。S2没有新增历史截断。
- 直接调用bare Qwen2底层Transformer最终hidden state，不请求每层hidden states；拒绝DDP/FSDP/PEFT等外层包装，绝不绕过.module。原IEEE完整Head、完整序列参考评分及Summary全词表熵路径保留。Reference无梯度、eval模式、实际激活重计算和上下文梯度均已检查。没有Sampled KL或目标token-logit近似。

## 性能口径与唯一优化器步骤

六条轨迹合计35160 native token、15222 action token。CPU预先按合法完成、首轮失败、后续失败三层各取最短与最长，共六条；不按数值或速度结果筛选。Actor使用已有32查询第一更新权重，Reference固定初始模型，分母和优势使用保存的真实初始行为值；这是**固定数据离线重放，不是新on-policy实验**。

同一双A800 allocation，S0/S1使用第一卡，S2使用第二卡；每候选内模型参数固定。一次预热后两次计时，CUDA同步；每批计时前清空allocator缓存并重置峰值，属于内核预热后的配对测试，不宣称生产环境热allocator吞吐。S1失败后不扩大，不跑S3，不增加其他精度或参数候选。

| 排他阶段，秒，均值 | S0 | S2 |
|---|---:|---:|
| 旧概率actor-only评分，wall |10.9180|10.4569|
| Actor forward，CUDA |10.7644|10.3199|
| Reference forward，CUDA |10.7611|10.3166|
| Exact KL，CUDA |0.1482|0.1470|
| backward，CUDA |32.8214|29.5927|
| PPO评分与完整梯度累积，wall |54.7054|50.5461|

完整细分见[HF_RUNTIME_BENCHMARK.csv](HF_RUNTIME_BENCHMARK.csv)。CUDA分段在PPO wall内部，不能再与它相加。旧概率评分只测actor forward，不覆盖保存的真实采样分母。所有测速重复均无optimizer.step。

按用户要求，最快且数值正确的S2另执行**一次**离线PPO更新：6条序列、原固定优势、sequence_mean（每序列token损失求和，再按六序列平均）、原KL/LR/clip参数、全新FP32 AdamW。其PPO评分/梯度累积/clip/step合计 **52.4183秒**，不含单列旧概率评分、初始化或保存；不是重新采样的RICE整轮。

实际grad norm=3101.332031，正/负/零优势token=352/13539/1331；真正clipped surrogate比例=0.085403%。实际optimizer.step=1，1549355798个参数元素改变，最大变化=1.90734863e-06；AdamW状态均FP32。Actor保存并重新加载，**全部参数逐项一致**。该文件是Actor-only离线诊断，不冒充包含Adam/RNG的正式续训检查点。该步骤只验证更新链路，不放宽速度采用门槛。

## 完整RICE收益：仅估计

旧32查询独占wall=7614.265秒，HF=2207.683秒（28.9940%）。将本轮1.07574×的前置HF评分/累积加速乐观应用于整个旧HF阶段、假设其他成本不变，Amdahl估计整轮仅加速1.02084×，节省**2.0414%（约155.44秒）**。优化器、I/O等固定项不一定同幅提速，实际收益可能更低。**未测新完整RICE wall，不是端到端提速结果。**

## 执行与边界

| Slurm作业 | 实际状态 | 内容 |
|---|---|---|
|176665|COMPLETED/0|固定六轨迹与事前门限、阅读安装版Qwen2实现|
|176666|FAILED/1，预期RED|3项缺失功能错误，明确不是通过|
|176669|COMPLETED/0|3项针对性CPU测试通过，0失败/0跳过|
|176670|COMPLETED/0，11:35|两A800，S0/S1/S2与条件S3决策；S1数值失败不等于进程失败|
|176675|COMPLETED/0，1:47|单A800，唯一离线step及Actor保存/加载|
|176676|COMPLETED/0|CPU脱敏导出|
|176677|COMPLETED/0|结束后重新核对原轨迹/old概率与优势SHA，全部与事前冻结值一致|

CPU测试覆盖原生因果位置、全词表熵、loss及全参数梯度、工具上下文梯度、Reference冻结、实际checkpoint重计算、包装/枚举拒绝及精度异常恢复。GPU原始输出、参数梯度、模型与轨迹保留私有；公开仅源码、模板及聚合统计。执行代码指纹见FINAL_DECISION；发表版本的modeling模块说明文案有更新，功能AST与实测版本一致。生产hf_runtime的opt-in位置投影接线仅完成静态检查，本轮未跑新的在线Trainer。

脱敏实际命令：

```text
sbatch --gres=gpu:2 --cpus-per-task=16 --mem=192G --time=00:45:00 <private-wrapper>
python -m unittest discover -s <test-copy> -p test_action_projection.py -v
python -u scripts/hf_compute_driver.py --plan <frozen-private-plan> --output <new-private-output>
python -u scripts/hf_compute_benchmark.py --scheme S2 --step --plan <same-private-plan> --output <separate-private-output>
```

无模型下载、依赖安装、新生成、检索、奖励重算、熵选点、开发评估或32/128训练。最多同时2GPU，GPU allocation合计约0.41583GPU小时。旧实验和持有作业未动；仅更新现有PR2、不合并、不生成压缩包。默认训练继续IEEE；S2仅通过`hf_projection_mode=action_positions_v1`显式启用，默认full_sequence，Summary熵仍走保留的完整路径。HF TF32仅留独立benchmark scope，生产入口拒绝`hf_matmul_precision`，避免误用或静默忽略。全局验证字段不升级。

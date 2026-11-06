# 实际执行记录

这些是实际执行结果，静态 AST/JSON/差异检查不计作运行测试通过。各次测试范围重叠，不相加为独立测试数或全仓通过率。既有真实资产均显式使用本地冻结路径，offline 开关开启，没有自动下载。

| CPU Slurm 作业 | 实际测试数 | 通过 | 失败 | 错误 | 跳过 | 范围/解释 |
|---|---:|---:|---:|---:|---:|---|
| 174582 | 135 | 131 | 4 | 0 | 0 | 新协议及相关原有组件；3个测试Tokenizer缺EOS片段映射，1个旧异常类型契约不符 |
| 174583 | 24 | 24 | 0 | 0 | 0 | 修复fixture和MalformedGenerationError包装后，定向重跑生成控制器与rollout |
| 174591 | 28 | 28 | 0 | 0 | 0 | IO、梯度对照、训练计划、模型评分、HF API |
| 174602 | 17 | 17 | 0 | 0 | 0 | 原生UTF-8拒绝/恢复、mask、训练IO、失败终态 |
| 174605 | 8 | 8 | 0 | 0 | 0 | 批量候选、格式、训练IO |

本机最后定向运行格式3项和批量2项，均通过。首次批量测试命令缺少PYTHONPATH，发生1个loader导入错误、未执行测试；设置`PYTHONPATH=src`后运行成功。没有把该错误计为通过，也没有为修复它修改算法。

实际CPU命令（仅工作/报告路径脱敏）：

```bash
PYTHONPATH=<EXISTING_PYTEST_PACKAGES>:<SOURCE_V1>/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q -ra tests/test_two_stage_masks.py tests/test_two_stage_format.py \
 tests/test_two_stage_generator.py tests/test_failure_zero.py tests/test_rollout.py \
 tests/test_credit.py tests/test_modeling.py tests/test_objective.py tests/test_workflow.py \
 tests/test_hf_generation_compat.py tests/test_formats.py tests/test_body_coverage.py \
 tests/test_config_recipes.py --basetemp=<REPORT>/tmp -o cache_dir=<REPORT>/cache \
 --junitxml=<REPORT>/junit.xml
```

后续同一命令参数形式的实际测试文件列表：

- 174583：`test_two_stage_generator.py test_rollout.py`。
- 174591：`test_two_stage_training_io.py test_training_plan.py test_modeling.py test_hf_generation_compat.py`。
- 174602：`test_two_stage_format.py test_two_stage_masks.py test_two_stage_training_io.py test_failure_zero.py`。
- 174605：`test_batched_generation.py test_two_stage_format.py test_two_stage_training_io.py`。

原JUnit含内部路径/主机字段，保留在独立存储，仅公开计数和SHA；不是上传原始JUnit。实际控制脚本哈希见 CONTROL_SOURCE_INDEX.json，相关JUnit及轨迹/优势/检查点记录哈希见 ARTIFACT_INDEX.json。

## 真实模型与跨组件验收

| 实际作业 | 结论 | 不能据此声称 |
|---|---|---|
| 174588 数值诊断 | BF16与BF16+FP32 head未通过预先声明门限；完整FP32两个固定短条件通过，未放宽门限 | BF16推理与FP32训练精确等价 |
| 174592 / 174606 | 匹配FP32训练八题 A=1/8、B=4/8五轮；B各轨迹完整原生历史LP门限通过 | 完整论文性能 |
| 174577 | 真实21,325,384,127字节同规模权重/优化器/RNG本地写入、加载、共享校验发布通过，无新step | 已修复旧损坏根因，或这是新的RICE训练checkpoint |
| 174607 / 174608 | 短串行/批量概率与行顺序通过；首次梯度诊断误用冻结评估模型而失败。解冻诊断参数后，真实1.5B直接/重计算loss及全参数grad相同；实际更换权重后批量概率通过 | 重新训练过模型、vLLM同步已验证 |
| 174632 | 四条身份预定真实前缀全部进入Summary，4/4合法；1,806真实token，完整历史概率通过 | 批量Trainer或完整五轮吞吐已验证 |
| 174610 / 174611 | 各一次真实在线更新，首批16条原生ID/mask/奖励/行为身份完全相同。RICE9个可估计、4个非零局部优势，实际梯度差和参数变化；完整32题开发 | 性能提高；观察到真实推理传播门控激活 |
| 174628 / 174643 | 两个实跑检查点独立进程恢复模型、完整优化器及CPU/CUDA RNG实际抽样哈希，step=1且没有新step | 可以忽略源码/数据/检索绑定跨版本恢复 |

开发结果均为负向性能结果：初始0.15928929、GRPO0.04709593、RICE0.02771415（完整32题失败调整任务效用）。正常轨迹NDCG及覆盖另列，不能以成功子集均值替代完整分数。

### 失败尝试未隐藏

- 174585：诊断脚本误用chat-template返回结构，生成前失败；后用已存在的renderer。
- 174586：BF16概率门限失败，未继续八题扫描。
- 174593：已付费真实Summary含无效UTF-8，原生证据保留；174594离线诊断后，174606精确复用17个动作，非新独立样本。
- 174604：恢复脚本将JSON列表与内存tuple直接比较，身份校验保守拒绝；改为稳定规范指纹，未改实际策略。
- 174629：单模型诊断误用要求两个可见设备的评估入口，初始化拒绝、未生成；174632改用明确的单模型诊断加载，生产双设备约束保持。
- 所有以上GPU分配秒数、失败生成和重复诊断均进入实际预算及token账本。

### 真实命令与绑定

```bash
sbatch --output=<LOG_ROOT>/two-stage-online-ricepo-%j.out \
 --error=<LOG_ROOT>/two-stage-online-ricepo-%j.err <CONTROL>/93-two-stage-train.sbatch ricepo
sbatch --output=<LOG_ROOT>/two-stage-online-grpo-%j.out \
 --error=<LOG_ROOT>/two-stage-online-grpo-%j.err <CONTROL>/93-two-stage-train.sbatch grpo
```

两个实际脚本各申请1节点、1进程、2GPU、16CPU、192G、4小时，分区/账户在公开材料中脱敏；实际运行仅一次更新。配置、全部解析参数及来源、代码指纹和产物哈希均已记录。后续没有正式训练提交。

两个真实独立恢复作业实际执行的命令形式：

```bash
PYTHONPATH=<EXACT_RECORDED_SOURCE>/src python -u <CHECK_TOOL>/verify_training_checkpoint.py \
 --run-root <TRAIN_RUN> --output <NEW_REPORT>.json --device cuda:0
```

使用绑定的精确包源码，对应公开等价提交98267806afc70bf73c26385460918d918b02ea49。核验工具来自当前`scripts/validation/verify_training_checkpoint.py`；提交作者信息脱敏不改变文件树或运行指纹。恢复不会重做optimizer.step。

## 未运行

四rank/FSDP等价性、vLLM推理副本概率与权重同步、正式128查询批次、完整十二域、正式测试集、32K极限训练容量、额外多次更新与跨种子统计均未运行，不记为跳过或通过。公开execution/runtime_validated/four_gpu_validated模板标志保持false。

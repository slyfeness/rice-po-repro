# 本阶段源码修改

| 文件 / 函数 | 修改与类别 | 算法语义 | 验证 |
|---|---|---|---|
| contracts.generation_action_mask / trajectory_behavior_log_probs / generation_from_record | 明确混合动作mask、阶段、标量概率及原生恢复；缺失工程能力 | 显式新环境协议；legacy默认保持 | CPU、真实轨迹及完整训练 |
| two_stage.TwoStageFormat / generate_two_stage | 原生两阶段、一次RNG流、共享4096预算、增量停止、先存证据；独立实现选择 | 环境插入与结束协议改变轨迹历史，非作者确认；不重加权条件logits | 固定8题、实际概率和新分支 |
| rollout / credit / workflow / failure_zero 的mask消费 | 替换全True假设、严格保留失败组；确定校验缺陷 | 信用公式/门控/奖励规则不改 | CPU及16原轨迹+80分支 |
| hf_runtime.HFNativeGenerator / update | 捕获实际行为分母，teacher数值门限，显式Actor精度；工程与数值实现 | FP32明示；T1/p1/k0保持 | BF16失败、FP32通过，真实更新 |
| modeling.enable_eval_layer_checkpointing | eval层重计算；可选优化 | 不detach上下文，不改loss | 真实短序列loss/grad一致及实际重计算 |
| training.ppo_plan_update / credit_gradient.prepare_comparison | 步前/后诊断，同原始GRPO反向但不step；工程诊断 | 正式PPO计划保持 | 梯度差与参数方向实测 |
| checkpoint_io / hf_runtime.finish | 本地验证、共享校验、原子发布；确定IO风险保护 | 不改学习目标 | 21GB同规模测试及两个真实检查点恢复 |
| holdout._actor_runtime | 统一实际Actor精度；工程一致性 | 公共评估与训练后选择约束保持 | 初始与两方法完整32题 |
| probability_check.compare_log_probs | 标量误差、实际ratio；诊断 | 不改概率 | 短条件及16轨迹完整历史 |
| batched_generation.generate_many | 独立逐请求RNG、原生left-pad/KV/概率；可选加速候选 | 相同无约束条件概率；未接入训练器 | CPU、真前缀、行顺序、真权重切换 |
| scripts/validation/verify_training_checkpoint.py | 源码绑定的独立模型/优化器/RNG恢复；工程能力 | 不执行新step | 两个真实GPU恢复作业 |
| configs/two_stage_* / deployment.two_stage.example.json | 明确变体、冻结配方；来源记录 | 保留paper.json，公开禁执行 | 实跑完整解析配置与来源已记录 |

严格解析器主体、objective.py、paper.json 和 compute_credit / build_advantages 的公式保持；credit的变化仅为动作mask校验。没有长度奖励、成本奖励、EOS屏蔽、重采直到成功、删除重复样本或失败分支。实跑源码与当前补充日志字段的范围分开，原始历史结果不覆盖。

# 检索与资产身份状态

实际执行的真实模型仅为策略模型 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，声明 revision 为 `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。`asset_identity.local_model_identity` 已计算本地权重、config、tokenizer 等文件内容指纹；聚合值与文件摘要见 [evidence.json](evidence.json)。本地报告 `upstream_verification=not_verified`，未独立认证这些本地文件来自该上游 revision。

实际 Diver 查询编码器与文档编码器的一致性为**未运行**，上游资产来源核验也为**未运行**。相同名称、revision 字符串或向量维度不能代替内容身份验证。

声明的后续检索目标为 `AQ-MedAI/Diver-Retriever-4B`，revision `ff7f8bc8d1827734dcf2bbfab99f065be618069a`；上游参考代码提交为 `cfdf09ad863540f78a655880ef133c4cb28b3676`。它们沿用 `provenance/sources.json` 的冻结信息，本阶段没有下载或核验真实检索权重。

| 检查 | 状态 | 现有证据 / 限制 |
|---|---|---|
| 模型实际指纹、旧 receipt 拒绝、行序错位 | 合成 CPU tests 已执行 | `tests/test_asset_binding.py`；不能代替真实检索资产核验 |
| 向量、行 ID、receipt 目录发布 | CPU fixtures 在目标挂载点执行 | `preparation._publish` 兼容补丁；跨节点并发未验证 |
| 真实编码输入与向量对齐 | 未运行 | 无版本 2 完整向量资产包 |
| 稠密/BM25 得分、归一化、混合、并列排名 | 未运行真实对比 | `retrieval.py`、`reference_compare.py` 仅有合成测试 |
| 固定 excluded_ids、NDCG@10 | 合成 CPU tests 已执行；真实集合未运行 | 没有正式检索排名、真实 NDCG 或基准成绩 |

未来必须冻结 summary_only / original_and_summary、截断、pooling、归一化、检索深度与展示数量等实际配方，逐层比较输入→向量→分数→排名→NDCG。上游行为与独立实现选择的区分见 [../../RETRIEVAL_AUDIT.md](../../RETRIEVAL_AUDIT.md)。此 PR 不宣称已恢复作者唯一实现。

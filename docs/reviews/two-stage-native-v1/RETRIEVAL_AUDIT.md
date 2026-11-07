# 固定检索与真实评分

没有重建索引、重编码语料或更换检索模型。三组使用同一 Diver-4B + BM25、语料向量及行ID资产，严格 load_vector_bundle 核对实际模型/Tokenizer/编码配方与receipt；本地一致性不自动升级上游版本真实性状态。

本次统一检索后端指纹为 `22e89e0218398dcbd406c1e370bf61af77ea74ae058154bca017e66423eb7aac`，开发 D0 NDCG@10 均为0.6343234049838116。查询组成保持 summary_only；原查询+摘要仍是另名敏感性配置，没有根据结果静默切换。检索深度、观察数量、截断和融合选择详见 EVIDENCE.resolved_parameters 及[原检索审查](../first-ricepo-smoke/RETRIEVAL_AUDIT.md)。

PyTrecReward 使用相同真实 qrels 与实际排名，原始NDCG保持[0,1]。可估计锚点的六个实际局部奖励、总体方差、最终效用及残差已独立复算，与训练优势一致。9条分支在锚点没有合法Summary，局部奖励仍为null，不补造检索或NDCG=0。后续失败的终态0是训练环境约定，不伪称实际排名NDCG。完整逐题D0/最后真实NDCG/失败效用及原始产物SHA均在索引；原始排名和qrels不公开。

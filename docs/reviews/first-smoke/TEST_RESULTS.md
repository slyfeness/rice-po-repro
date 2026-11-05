# 实际测试与未运行范围

以下是已发生的授权试跑，不是本轮报告整理时重新执行。统计来自私有日志和 JUnit 的本地副本；本轮用标准库重新解析 JUnit 并核对文本摘要，结果及原件 SHA-256 见 [evidence.json](evidence.json)。不提供把跳过算作通过的通过率。

| 作业别名 | 状态 | 通过 | 失败 | 跳过 | 运行时间与证据 |
|---|---|---:|---:|---:|---|
| CPU-ENV-01 | 启动失败，环境清单完成 | — | — | — | batch 46 秒；主环境缺 pytest；ENV-REPORT-01 |
| CPU-LAUNCH-02 | 启动失败，未收集测试 | — | — | — | batch 1 秒；复用路径缺 py.py；早期试跑记录，未附原日志副本 |
| CPU-TEST-03 | 失败 | 217 | 2 | 2 | pytest 77.19 秒、batch 80 秒；CPU-LOG-03 |
| CPU-TEST-04 | 已通过所执行测试，仍有跳过 | 219 | 0 | 2 | pytest 18.20 秒、batch 21 秒；CPU-LOG-04、CPU-JUNIT-04 |
| GPU-PROBE-01 | 短数值诊断已通过断言 | — | — | — | batch 135 秒；GPU-REPORT-01；不是 pytest case 统计 |

历史 Slurm 终态与 batch 耗时在试跑时核对。当前可公开索引未包含单独原始 sacct 导出，不能仅凭这些 JSON 摘要独立认证调度器终态。排队耗时没有重新构造，不将 batch RUNNING 状态当作模型就绪。GPU 模型初始化与内容哈希共 71.8434 秒；框架导入后探针总计 78.3347 秒，均不等于整个 batch 135 秒。

## 失败、修复与跳过

CPU-TEST-03 两项失败：

- `tests/test_asset_binding.py::test_atomic_bundle_and_strict_load_detect_changed_weights_order_and_old_receipt`
- `tests/test_asset_binding.py::test_failed_publish_or_encode_never_exposes_a_partial_bundle`

均由 `preparation._publish` 的 `renameat2(RENAME_NOREPLACE)` 在挂载点返回 EINVAL 引起。应用 FIX-PATCH-01 后，完整 CPU suite 的后续运行记录为 219 通过、2 跳过。相应代码提交为 `6f87858`；本 PR 不另改算法源码。

两项跳过来自 `tests/test_opt_in_assets.py`：未提供带独立 SHA-256 的 `RICEPO_TOKENIZER_FIXTURE` 和 `RICEPO_GPU_ENTROPY_FIXTURE`。不自动下载，不用独立 GPU 探针覆盖其跳过状态。每模块统计和完整跳过测试名称见 evidence.json。

## 真实模型探针

调用 `TransformersEntropyProvider.token_entropies`、`sampled_old_log_probs`、`causal_token_scores`，同一短输入上比较全词表熵、旧/当前采样 token 评分与固定参考 KL，然后反向传播。14 个 token 中末四个为评分位置；actor/reference 均 eval、FP32、cuda:0，reference 冻结。

| 检查 | 实测 | 实际断言 |
|---|---:|---|
| 全词表熵与完整前向最大绝对误差 | 0.000011563301086425781 | `allclose(atol=2e-4, rtol=2e-5)` |
| old/current 最大绝对误差 | 0 | `allclose(atol=1e-5, rtol=1e-5)` |
| 固定参考条件 KL 最大绝对值 | 0 | `<1e-5` |
| 反向梯度范数 | 57.766197204589844 | 有限且 >0；reference 所有 grad 为 None |
| optimizer step | 0 | 未创建/调用优化器更新 |

这里的 KL 是给定已采样历史的条件 KL；没有枚举所有可能轨迹。此梯度探针不是 PPO 学习有效性结论，也不要求每个真实奖励批次非零梯度。

## 实际命令的脱敏转录

下列命令来自已执行 batch 脚本的核心语句，变量值替代内部路径；不是本轮执行命令。提交与全部计算经 Slurm；不公开账户、partition、真实路径或地址。没有将这些内容接入 CI。原始 batch 与探针脚本的 SHA-256 见证据索引。

CPU-TEST-04 在已准备的只读 pytest 兼容导入路径中执行：

```bash
export PYTHONPATH="$PYTEST_COMPAT:$SOURCE_CHECKOUT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4
export TMPDIR="$PROJECT_ROOT/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
cd "$SOURCE_CHECKOUT"
patch -p1 --forward < "$PUBLICATION_FIX_PATCH"
"$PYTHON" -m pytest -q -ra --basetemp="$REPORT_ROOT/pytest-tmp" -o cache_dir="$REPORT_ROOT/pytest-cache" --junitxml="$REPORT_ROOT/junit.xml" > "$REPORT_ROOT/pytest.txt" 2>&1
```

GPU-PROBE-01：

```bash
export PYTHONPATH="$SOURCE_CHECKOUT/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12
export TMPDIR="$PROJECT_ROOT/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYTHON" "$ACTOR_PROBE_SOURCE"
```

路径参数化的探针文本附于 [actor-probe.py.txt](actor-probe.py.txt)。它是公开转录，使用 PROJECT_ROOT / REPORT_ROOT / LOCAL_ACTOR 环境变量替代原始路径，原件哈希为 GPU-SOURCE-01；不声称两文件逐字相同。输入、模型加载参数、容差、评分和反向逻辑保留。未公开部署值与授权配置。

## 测试分级与缺口

纯 Python / CPU 张量测试已按上述 suite 实际执行；真实本地 tokenizer 只参与独立模型探针，不等于 opt-in tokenizer fixture 通过。GPU 单模型短评分/梯度已诊断（同时持有冻结 reference）；GPU 原生 rollout、双设备端到端和分布式等价性均未运行。真实数据与检索对齐、完整查询集合、长上下文、真实分支与 checkpoint/开发恢复也未运行。

本 PR 的新增改动仅为文档与审核政策，未新增测试源码或重跑 suite。上传前仅做文本、JSON、探针文本 AST、哈希、差异与公开风险检查，其结果单列为静态检查。

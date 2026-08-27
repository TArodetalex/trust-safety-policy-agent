# Day 6 执行记录

## 已完成

- 新增版本化生产路由配置 `production_routing_v1.json`。
- 新增 `ProductionRouter`，实现 Rules、OCR、LLM、人工复核四级路由。
- 为生产 CLI 和 Golden Set 评测增加 `production` 引擎。
- 新增逐阶段路由审计文件 `routing_records.jsonl`。
- 评测报告新增 development、regression、blind 数据切片。
- 图片自动通过增加正向政策证据要求。
- 新增文本、OCR、LLM、远程图片、输入冲突和失败关闭测试。

## 离线验收

执行命令：

```bash
PYTHONPATH=src python3 scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v3.csv \
  --engine production \
  --run-id EV-DAY6-PRODUCTION-OFFLINE-V1 \
  --strict
```

结果：

- 150 条案例，122 条自动判定，28 条人工复核。
- 覆盖率 81.3%，分流准确率 98.0%。
- 自动判定准确率和政策准确率均为 100%。
- 误放率和误拒率均为 0%。
- 全部冻结质量门禁通过。
- 95 条由 Rules 完成，27 条由 Rules + OCR 完成。
- 盲测覆盖率 72.5%，但自动判定准确率仍为 100%。

## 未完成项

当前环境未配置 `LLM_API_KEY` 或 `OPENAI_API_KEY`，因此没有调用外部多模态
模型，也没有冻结 Day 6 LLM 基准。真实模型评测必须使用独立 run ID，保留
完整 `report.json`、`records.jsonl`、`errors.csv` 和
`routing_records.jsonl` 后再决定是否发布。

建议 run ID：

```text
EV-DAY6-PRODUCTION-LLM-V1
```

模型接入后的首要验收目标是提升 blind 和 multimodal 切片覆盖率，同时保持
零误放、零误拒。不得通过降低 0.85 置信度阈值来满足覆盖率指标。

## OpenRouter 探索性评测

OpenRouter Key 鉴权成功，但账户 `total_credits=0`，不能调用付费的
`openai/gpt-4o`。使用 `openrouter/free` 完成了真实远程多模态评测：

```text
EV-DAY6-PRODUCTION-LLM-V1
```

三个 OCR 未决图片均升级到远程模型。模型对两条案例给出高置信度普通通过，
但没有具体豁免；另一条返回不可验证结果。生产守卫将三条全部转人工。

最终指标与离线基准一致：

- 覆盖率 81.3%，分流准确率 98.0%。
- 自动判定准确率和政策准确率均为 100%。
- 误放率和误拒率均为 0%。
- blind 覆盖率 72.5%。

评测过程中曾发现免费模型把无品牌商品错误关联到二手豁免。路由器已增加
`visual-evidence-guard`：LLM 图片通过必须由确定性规则从模型观察文本中重新
命中具体豁免，不能仅凭模型选择的 chunk 自动通过。

由于 `openrouter/free` 会动态选择上游模型且存在共享池限流，本次结果不能冻结
为固定模型基准。充值后应使用固定的 `openai/gpt-4o` 重新运行。

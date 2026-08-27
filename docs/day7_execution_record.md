# Day 7 执行记录

## 目标

在不降低判定置信度阈值和安全守卫的前提下，补全 Golden Set v4、严格
多模态输出协议、固定模型 shadow 评测和版本化基准。

## 已完成

- 新增 `build_golden_set_v4.py`，可从冻结的 v3 数据确定性生成 v4。
- v4 包含 210 条案例、90 条多模态案例和 60 条 blind 案例。
- 新增 60 条 Day 7 视觉夹具，覆盖五类违规、五类豁免、干净无品牌负向
  对照、低对比度、旋转、小字和遮挡。
- 数据校验器可同时验证 v3 与 v4 资产清单及 SHA-256。
- 新增 v4 配额计划、切分清单、标注台账和治理测试。
- 多模态请求由 `json_object` 升级为 Pydantic 生成的严格
  `json_schema`，所有字段必填且禁止额外字段。
- OpenRouter 请求默认要求 provider 支持全部参数并禁止 fallback，可通过
  `LLM_PROVIDER_ORDER` 固定 provider。
- 新增全图片 `--shadow-multimodal` 模式。生产决策不受 shadow 输出影响。
- shadow 记录实际模型、provider、请求 ID、Schema 状态、tokens、费用、
  延迟、决策、政策标签和证据 chunk。
- 新增独立 shadow 门禁，要求单一 provider、单一模型、Schema 成功率、
  决策质量和零误放/误拒。

## 离线基准

```text
EV-DAY7-PRODUCTION-OFFLINE-V1
```

Golden Set v4 离线结果：

- 210 条案例，162 条自动判定，48 条人工复核。
- 自动判定准确率 100%，政策准确率 100%。
- 覆盖率 77.1%，分流准确率 98.6%。
- 误放率 0%，误拒率 0%。
- Day 7 新增切片 60 条，自动判定准确率和政策准确率均为 100%。
- 干净无品牌图片仍全部进入人工复核，符合 shadow-first 约束。

Day 7 数据包含 45 条预期人工复核案例，理论最大覆盖率为 78.6%。因此离线
门禁将覆盖率设为 75%，同时将自动准确率提高到 98%、分流准确率提高到 95%、
政策准确率提高到 98%，并要求误放和误拒均为 0。该调整反映数据集路由构成，
没有降低任何引擎置信度阈值。

## 固定模型 Shadow

正式 shadow 运行需要已充值账户和未暴露的新 Key：

```bash
export LLM_API_KEY="<NEW_KEY>"
export LLM_API_BASE="https://openrouter.ai/api/v1"
export LLM_MODEL="openai/gpt-4o"
export LLM_PROVIDER_ORDER="OpenAI"
export LLM_REQUIRE_PARAMETERS="true"
export LLM_ALLOW_FALLBACKS="false"
export LLM_TIMEOUT_SECONDS="120"

PYTHONPATH=src python3 scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v4.csv \
  --engine production \
  --gates config/evaluation_gates_day7_v1.json \
  --shadow-multimodal \
  --run-id EV-DAY7-GPT4O-SHADOW-V1 \
  --strict
```

成功运行后必须保留 `report.json`、`records.jsonl`、
`routing_records.jsonl`、`shadow_records.jsonl` 和
`shadow_report.json`。只有生产门禁和 shadow 门禁同时通过，固定模型结果
才可冻结为 Day 7 远程基准。

当前仓库未保存任何 API Key。真实固定模型 shadow 报告仍等待外部凭证和额度，
不能由离线结果替代。

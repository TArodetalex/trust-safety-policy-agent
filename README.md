# Trust & Safety AI Decision Lab

一个面向平台审核人员与策略运营人员的 AI 审核辅助工作台。项目以抽象化的平台治理场景为背景，将商家身份审核、商品知识产权审核、政策检索、人工复核和质量评测整合到同一套可运行流程中。

本项目是个人作品集 Demo，使用公开品牌信息、合成政策和合成案例，不包含任何公司内部政策、生产数据或真实用户信息。系统只提供风险建议与证据，不直接执行下架、封禁等平台处置。

## 项目要解决的问题

- 前置机审证据不足时，部分 Case 无法进入正式审核流程。
- 简单审核任务仍需跨页面补充字段，操作链路较长。
- 政策、品牌知识、Prompt 和审核结论分散，难以持续维护和追溯。
- 模型结果缺少人工复核、问题归因和版本回退机制。

项目将这些问题收敛为一套 AI 辅助审核产品：统一接收 Case 证据，自动路由商家或商品 Workflow，结合政策与品牌知识生成结构化建议，并在证据不足或结果冲突时转交人工。

## 核心能力

### 审核员工作台

- **综合审核**：识别 Shop/Product Case，并拆分为独立审核任务。
- **商家侧审核**：分别判断店铺名称与头像风险，校验品牌授权和 Meaningful Word 豁免。
- **商品侧审核**：分析 Counterfeit、Knockoff、MBA 与 TMI，支持合法的多标签组合。
- **我的 Prompt**：保存个人 Prompt 版本，对同一 Case 运行实验分析，不覆盖正式结果。
- **Case 管理**：支持新增、修订、归档以及 CSV/Excel 导入导出。

### 策略与 AI 运营工作台

- 维护正式 Workflow、Prompt、Skill 及其发布版本。
- 分别维护政策知识库与品牌商品知识库。
- 查看 Agent Decision、Reviewer Decision 和 Final Decision。
- 通过 Trace、Bad Case 和 Replay 定位问题首次发生的节点。
- 使用固定数据与配置比较候选版本，保留发布和回退记录。

### AI 与治理能力

- 接入 OpenAI-compatible 多模态模型 API，并支持阿里云百炼模型配置。
- Policy RAG 与 Brand/Product RAG 使用独立索引和版本。
- Prompt、Skill、模型、Workflow 和知识库版本随运行记录保存。
- 受控 Tool Calling 使用工具白名单、参数校验、步数限制和结果 Guardrail。
- API Key、Authorization、Cookie、Token 和图片二进制不会写入 Trace。
- Counterfeit 与 Knockoff 冲突、图片不可读或关键证据缺失时自动转人工。

## 审核流程

正式 Shop/Product Workflow 均在执行过程中记录节点，而不是在结果生成后拼接摘要：

```text
输入校验
  -> 证据标准化与提取
  -> 品牌候选召回
  -> 政策检索
  -> 品牌知识检索
  -> 规则与模型分析
  -> 结果聚合
  -> Guardrail
  -> 发布审核建议
```

每次运行生成独立的 Agent Run，并保存 Case 修订、Workflow/Prompt/Skill 版本、模型、索引版本、节点状态、延迟和脱敏错误。人工结论以 Reviewer Annotation 单独保存，不覆盖模型历史结果。

## 数据与知识库

- `synthetic_policy_v2`：结构化描述 Counterfeit、Knockoff、MBA、TMI、Shop Identity、豁免和转人工条件。
- Controlled Brand Library：包含 50 个公共管控品牌，并与合成测试品牌分开标识。
- Brand/Product Knowledge：保存品牌标准名、别名、拼写变体、品类和代表产品。
- Demo Case Store：提供 10 条 Shop Case 和 18 条 Product Case，覆盖正常、违规、授权、豁免、证据不足和组合风险。
- Golden Set v4：保留 210 条合成历史回归案例，用于验证既有规则和流程稳定性；不会被日常 Case 或候选数据覆盖。

项目中的商品图为合成演示素材，并带有来源说明与 SHA-256 记录。离线规则不会假装从像素中识别视觉事实；真实图片理解仅在配置多模态模型后启用。

## 本地运行

### Python

需要 Python 3.10 或更高版本：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
streamlit run app.py
```

打开 `http://localhost:8501`。

### Docker

无需 API Key 也可以运行离线规则、知识检索和演示流程：

```bash
docker compose up --build -d
```

查看日志：

```bash
docker compose logs -f policy-agent
```

停止服务：

```bash
docker compose down
```

可以通过 `APP_PORT` 修改宿主机端口。

## 模型配置

复制 `.env.example` 为 `.env`，填入自己的模型服务配置：

```env
LLM_API_KEY=your_private_key
LLM_API_BASE=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your_multimodal_model
```

`.env` 已被 Git 忽略。请勿把真实 API Key 写入代码、测试数据、README 或 Git 历史。

也可以在应用的“模型与系统设置”中为当前会话临时配置 API 地址、Key 和模型。页面配置不会改写本地 `.env` 文件，重新启动会话后仍以环境变量为准。

阿里云百炼的配置说明与安全使用方式见 [`docs/bailian_setup.md`](docs/bailian_setup.md)。

## 测试与评测

运行自动化测试：

```bash
pytest
```

验证合成评测数据与历史回归数据：

```bash
python scripts/validate_golden_set.py
python scripts/verify_baseline.py
```

自动化测试主要验证数据模型、规则组合、授权逻辑、RAG 隔离、价格工具、Trace、Replay、Prompt 版本、发布回退和敏感信息过滤。它们用于证明系统行为符合预期，不等同于真实线上准确率。

模型效果评测需要固定 Dataset、Workflow、Prompt、Skill、索引、温度和输出 Schema，仅改变待比较变量。没有完成真实模型批量评测时，项目不会宣称生产准确率或业务提效。

## MCP 接口

项目提供可选的 MCP 服务，用于把政策检索、品牌查询、信号分析和审核能力暴露为结构化工具。MCP 不是运行 Web 工作台的必要条件。

验证本地 MCP 握手与工具调用：

```bash
python scripts/smoke_mcp.py
```

启动可选的 HTTP 服务：

```bash
docker compose --profile mcp up --build -d policy-mcp
```

默认端点为 `http://localhost:8000/mcp`。写操作同时要求调用方权限和显式人工确认。

## 主要目录

```text
trust-safety-policy-agent/
├── app.py                         # Streamlit 应用入口
├── config/                        # Prompt、Skill 与评测配置
├── data/
│   ├── brands/                    # 受控品牌库
│   ├── case_store/                # 合成演示 Case
│   ├── demo_assets/               # 合成商品图片及清单
│   ├── golden_set/                # 历史回归集与候选集
│   ├── knowledge/                 # 品牌商品知识源文档
│   └── policies/                  # 合成政策源文档
├── docs/                          # 使用说明与实现证据
├── scripts/                       # 数据校验、评测与 smoke test
├── src/trust_safety_agent/        # 核心业务与 AI 模块
├── tests/                         # 自动化测试
├── compose.yaml
└── pyproject.toml
```

## 能力边界

- 项目是本地可运行的作品集原型，并非企业生产系统。
- 数据、政策和图片均为公开信息或合成内容，不代表任何公司的内部规则。
- 离线 Provider 用于可复现演示，不伪装成实时网络检索结果。
- Live Reference Price Provider 只保留明确的接入边界；无可靠来源时不会确认 Counterfeit。
- Windows 和 Linux 使用配置的多模态模型处理图片；macOS 可额外使用本地 Vision OCR helper。
- 系统输出是审核建议，最终结论仍需人工确认。

## License

本仓库用于个人学习、作品展示与技术交流。使用其中的合成数据和规则时，请保留其非生产、非公司内部资料的边界说明。

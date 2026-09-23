# Trust & Safety AI Decision Lab

A controlled AI decision, evaluation, and MCP interoperability lab grounded in
a Trust & Safety use case. The project demonstrates how a product workflow can
combine policy retrieval, model-selected tools, deterministic guardrails,
human review, and reproducible evaluation rather than stopping at a chat UI.

## Current Status

- Versioned data contract with backward-compatible v1 loading
- Synthetic policy covering five violations and five exemptions
- Golden Set v4 with 210 cases, 60 blind cases, and 90 multimodal fixtures
- Structure-aware Markdown policy parsing with stable chunk IDs
- Offline reproducible embeddings and persistent Chroma retrieval
- Policy-grounded single-case adjudication with evidence and review fallback
- Controlled Brand Library and deterministic Product IPR review pipeline
- Shop Identity review with authorization-first gating
- Validated Product/Shop CSV and Excel batch import and export
- Reviewer Workspace with separate Agent, Reviewer, and Final decisions
- Golden Set staging plus explicit v5 promotion and SHA-256 manifest
- Structured redacted traces and deterministic failure taxonomy
- Strict JSON Schema multimodal inference with provider telemetry
- Chinese-first Streamlit workspace with case, product, retrieval, and chunk views
- Alibaba Cloud Bailian fallback, redacted errors, metadata, and smoke testing
- Bounded schema-driven Tool Calling Agent with allowlists and step limits
- Versioned Prompt & Skill Studio with editable prompts and runtime capability bundles
- AI Evaluation Lab for reproducible run creation and baseline/candidate diffs
- Official MCP Python SDK server with tools, resources, prompt, and two transports
- Evaluation reports, confusion matrices, slices, error buckets, and quality gates
- Frozen Day 4-7 baselines and versioned dataset quota governance

See
[`docs/project_progress_optimization_report.md`](docs/project_progress_optimization_report.md)
for the current delivery assessment, production gaps, priorities, and roadmap.
See [`docs/phase0_phase1_execution_report.md`](docs/phase0_phase1_execution_report.md)
for the verified Phase 0/API/Phase 1 implementation record and claim boundary.
See [`docs/phase2_phase5_evidence_report.md`](docs/phase2_phase5_evidence_report.md)
for the Phase 2-5 workflows, examples, verification evidence, and limitations.

## Project Layout

```text
trust-safety-policy-agent/
├── app.py                         # Streamlit entry point (Day 2+)
├── assets/                        # Screenshots and architecture diagrams
├── data/
│   ├── baselines/                 # Frozen baseline manifests
│   ├── brands/                    # Versioned controlled-brand library
│   ├── examples/                  # Product and Shop batch examples
│   ├── eval_runs/                 # Generated evaluation outputs
│   ├── golden_set/
│   │   ├── golden_set_v1.csv
│   │   ├── golden_set_v2.csv
│   │   ├── golden_set_v3.csv
│   │   ├── golden_set_v4.csv
│   │   ├── assets/v3|v4/          # Hashed controlled image fixtures
│   │   ├── splits/v3|v4/          # Tuning and blind views
│   │   └── plans/                 # Versioned quotas and audit templates
│   └── policies/
│       └── mock_policy_v1.md
├── docs/
│   ├── annotation_guideline.md
│   └── data_contract.md
├── scripts/
│   ├── audit_dataset_plan.py
│   ├── run_evaluation.py
│   ├── validate_golden_set.py
│   └── verify_baseline.py
├── src/trust_safety_agent/
│   ├── __init__.py
│   ├── config.py
│   ├── adjudicator.py
│   ├── brand_library.py
│   ├── product_review.py
│   ├── shop_identity.py
│   ├── batch_io.py
│   ├── reviewer_workspace.py
│   ├── staging_dataset.py
│   ├── trace.py
│   ├── trace_builders.py
│   ├── failure_taxonomy.py
│   ├── controlled_tools.py
│   ├── controlled_agent.py
│   ├── prompt_skills.py
│   ├── evaluation_lab.py
│   ├── mcp_server.py
│   ├── llm_adjudicator.py
│   ├── llm_client.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── policy_loader.py
│   ├── vector_store.py
│   └── schema.py
├── tests/
│   ├── test_policy_loader.py
│   ├── test_retrieval.py
│   └── test_schema.py
└── pyproject.toml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Docker

Start the web application without an API key. Offline rules and policy
retrieval remain available:

```bash
docker compose up --build -d
```

Open `http://localhost:8501`. The Chroma index is persisted in the
`policy-chroma` Docker volume. Reviewer records, staging candidates, and traces
use separate Docker volumes and are not committed to Git.

For multimodal inference, copy `.env.example` to `.env`, configure a new API
key and model, then restart:

```bash
docker compose up --build -d
docker compose logs -f policy-agent
```

For Alibaba Cloud Model Studio (Bailian), including safe local key handling,
structured-output compatibility, controlled model fallback, and the one-call
smoke test, see [`docs/bailian_setup.md`](docs/bailian_setup.md).

Use `APP_PORT` to expose a different host port:

```bash
APP_PORT=8080 docker compose up -d
```

Stop the service with `docker compose down`. Add `--volumes` only when the
persisted policy index should also be deleted.

## Platform Compatibility

Text rules, policy retrieval, evaluation, and the Streamlit workspace run on
Windows, macOS, and Linux. The bundled local OCR helper uses Apple's Vision
framework and therefore runs only on macOS. On Windows and in the Linux Docker
image, image cases continue to the configured multimodal model; without a
configured model they fail closed to human review. This changes offline image
coverage, not the safety contract.

Use a multimodal provider for image review on Windows. Keep the API key in an
environment variable or an ignored `.env` file; never commit it. Frozen
baseline verification normalizes text line endings so LF and CRLF checkouts
produce the same artifact hashes.

## Build the Policy KB

```bash
python scripts/build_policy_index.py \
  --query "Gucci 1:1 replica handbag" \
  --top-k 3
```

The persistent index defaults to `data/chroma` in the project workspace. Set
`POLICY_DB_PATH` to override it.

## Run a Case Decision

Configure an OpenAI-compatible vision model:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
```

Run multimodal inference with a local image:

```bash
python scripts/adjudicate_case.py \
  --engine llm \
  --content-type product \
  --image ./product.jpg \
  "Seller caption and listing context"
```

The Streamlit sidebar accepts the same API settings for the current browser
session. Uploaded images are sent to that configured endpoint and are not
persisted by this project.

## Run a Product IPR Review

Open the `商品知识产权审核` tab in the Streamlit app. This workflow uses the
versioned Controlled Brand Library in `data/brands/controlled_brands_v1.csv`
and keeps these stages independently inspectable:

1. Candidate recall from the seller brand field, title, description, OCR text,
   and supplied visual marks.
2. Boundary-safe matching against controlled brand names and aliases.
3. Context validation for counterfeit language, compatibility, second-hand
   sales, and ambiguous common-word brands.
4. Policy retrieval and an auditable `approve`, `reject`, or `manual_review`
   recommendation with reviewer checkpoints.

A brand match is not treated as proof of infringement. Image submissions that
do not include reliable OCR or visual marks route to manual review.

## Batch and Human Review

The Chinese-first Streamlit workspace includes dedicated views for Product IPR,
Shop Identity, batch review, human review, traces, policy retrieval, and policy
chunks. Batch review accepts UTF-8 CSV or `.xlsx`, validates required columns,
URLs, duplicate IDs, empty files, and malformed rows, then exports filtered
results as CSV or Excel.

Example inputs:

- `data/examples/product_review_batch.csv`
- `data/examples/shop_identity_batch.csv`

Shop authorization is evaluated before identity risk. A verified `authorized`
status produces an authorization exemption; `unknown` is never treated as
authorized. Avatar URLs are not treated as visual evidence unless reliable
visual marks are supplied.

Reviewer records preserve `agent_decision`, `reviewer_label`, and
`final_decision` separately. Human-confirmed cases may enter the staging
dataset, but the frozen Golden Set v4 is never modified. An explicit Promote
action creates a separate v5 JSONL file and SHA-256 manifest.

Execution traces record workflow nodes, sanitized summaries, outputs, status,
latency, optional token usage, confidence, and errors. Keys, tokens, passwords,
credentials, and Bearer values are recursively redacted before a trace is
saved. New evaluation runs include `failure_counts` and a separate
`failure_report.json`.

## Controlled Agent

The `受控 Agent` workspace uses the configured model as a planner. On each turn
the model must either select one typed tool or return a final decision. The host
enforces a tool allowlist, Pydantic input validation, a four-step limit,
duplicate-call blocking, read/write separation, and a final evidence guardrail.
The model cannot directly execute arbitrary code or write to the review queue.

The planner accepts a narrow set of common OpenAI-compatible JSON tool-call
shapes and normalizes them before strict validation. A reject is allowed only
when a successful tool returned the same policy ID. Low-confidence, malformed,
repeated, unauthorized, or unevidenced actions fail closed to human review.

### How a knockoff decision is made

The demo deliberately keeps three layers separate:

1. The offline baseline uses explicit, testable signals. A knockoff candidate
   requires an imitation phrase such as `dupe`, `knockoff`, `inspired copy`, or
   `same design as`, together with a protected-brand reference. It then grounds
   the result in `POL-KO-001`; a brand mention alone is not enough.
2. The multimodal case workflow sends retrieved policy chunks plus a versioned
   Prompt to the configured model. Code validates the returned schema, policy
   evidence, exemption, and confidence before accepting the decision.
3. The controlled Agent selects typed tools such as
   `classify_policy_signals`, while the selected Skill restricts its Prompt,
   tool allowlist, policy scope, maximum steps, and output contract.

The `Prompt 与 Skill` workspace displays all built-in versions and can append
custom versions to ignored local JSONL files. Temporary Prompt edits affect
only that browser submission. Product IPR and Shop Identity remain
deterministic pipelines; they do not expose decorative Prompt fields because
those workflows do not call a model.

Run one live call without printing the key:

```bash
python scripts/smoke_controlled_agent.py
```

## AI Evaluation Lab

The `评测实验室` workspace indexes versioned evaluation artifacts under
`data/eval_runs`. It can launch a new deterministic Golden Set v4 rules run and
compare any baseline/candidate pair across quality, routing, review rate,
failed gates, tokens, cost, and latency when those telemetry fields exist.

This separates a model or Prompt change from a release decision: every result
keeps its engine, Prompt version, retrieval version, dataset fingerprint, and
quality-gate outcome.

## MCP Integration

`trust_safety_agent.mcp_server` is built with the official MCP Python SDK. It
publishes six tools (`search_policy`, `lookup_brand`,
`classify_policy_signals`, `review_product`, `get_review_queue`, and guarded
`submit_human_review`), two resources, and one review prompt. Type hints become
protocol schemas and successful tool results are returned as structured MCP
content.

Verify a real stdio subprocess handshake, discovery, resource listing, and tool
call:

```bash
python scripts/smoke_mcp.py
```

Start the optional Streamable HTTP service with Docker:

```bash
docker compose --profile mcp up --build -d policy-mcp
```

The endpoint is `http://localhost:8000/mcp` by default. The write tool requires
both caller-side write permission and an explicit `user_confirmed=true`; read
tools never receive the LLM API key.

Run the deterministic offline fallback:

```bash
python scripts/adjudicate_case.py \
  --engine rules \
  --content-type product \
  "Gucci 1:1 replica handbag"
```

Run the deterministic Day 3 regression smoke test:

```bash
python scripts/run_golden_smoke.py --strict
```

## Run the Evaluation

Run the Day 4 offline baseline with quality gates:

```bash
python scripts/run_evaluation.py \
  --engine rules \
  --run-id EV-DAY4-RULES-V1 \
  --strict
```

Each run writes `report.json`, `records.jsonl`, and `errors.csv` under
`data/eval_runs/<run-id>/`. Reports include strict accuracy, auto-decision
accuracy, coverage, routing accuracy, policy accuracy, reject precision/recall,
false approve/reject rates, confusion matrices, and metadata slices.

Use `--engine llm` after configuring the multimodal provider. All evaluation
engines use the same dataset, metrics, and gates.

On macOS, the local `rules-ocr` engine uses Apple Vision OCR and then applies
the deterministic policy rules. It only automates explicit violations and
supported exemptions; unreadable images and negative visual claims stay in
review:

```bash
python scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v3.csv \
  --engine rules-ocr \
  --run-id EV-DAY5-RULES-OCR-V1 \
  --strict
```

Run the Day 6 production router. It uses deterministic rules for text, local
OCR for local images, an optional configured multimodal LLM for unresolved or
remote images, and fails closed to human review:

```bash
python scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v3.csv \
  --engine production \
  --run-id EV-DAY6-PRODUCTION-OFFLINE-V1 \
  --strict
```

Production runs also write `routing_records.jsonl` with the selected engine and
every escalation step. See `docs/day6_production_routing_strategy.md` for the
thresholds and release rules.

Run the Day 7 offline production baseline against Golden Set v4:

```bash
python scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v4.csv \
  --engine production \
  --gates config/evaluation_gates_day7_v1.json \
  --run-id EV-DAY7-PRODUCTION-OFFLINE-V1 \
  --strict
```

After configuring a fixed multimodal model and provider, add
`--shadow-multimodal`. Production decisions remain unchanged while every image
also produces `shadow_records.jsonl` and `shadow_report.json`:

```bash
python scripts/run_evaluation.py \
  --dataset data/golden_set/golden_set_v4.csv \
  --engine production \
  --gates config/evaluation_gates_day7_v1.json \
  --shadow-multimodal \
  --run-id EV-DAY7-GPT4O-SHADOW-V1 \
  --strict
```

The shadow gate requires strict-schema success, fixed provider/model metadata,
decision and policy quality, and zero false approvals or false rejections.

## Govern Dataset Expansion

Verify that the Day 4 baseline inputs and expected metrics have not changed:

```bash
python scripts/verify_baseline.py --require-report
python scripts/verify_baseline.py \
  --manifest data/baselines/day5_rules_ocr_v1.json \
  --require-report
python scripts/verify_baseline.py \
  --manifest data/baselines/day6_production_offline_v1.json \
  --require-report
python scripts/verify_baseline.py \
  --manifest data/baselines/day7_production_offline_v1.json \
  --require-report
```

Audit the current seed dataset against the Golden Set v3 targets:

```bash
python scripts/audit_dataset_plan.py
```

Golden Set v4 contains 210 cases, including 60 separately materialized blind
cases and 90 hashed multimodal fixtures. Rebuild and validate it with:

```bash
python scripts/build_golden_set_v4.py
python scripts/validate_golden_set.py
python scripts/audit_dataset_plan.py --strict
```

The generated annotation ledger uses `synthetic_verified`, which records policy
consistency but does not replace independent human signoff.

## Run the App

```bash
streamlit run app.py
```

## Validate the Project

```bash
python scripts/validate_golden_set.py
python scripts/verify_baseline.py
pytest
```

The embedding and offline adjudication baselines remain deterministic. The
multimodal engine uses a real OpenAI-compatible Chat Completions endpoint while
preserving the same Policy KB and output contracts. See
`docs/adjudication_baseline.md` for decision and evidence invariants.

The dataset is synthetic and does not reproduce private or production policy
content.

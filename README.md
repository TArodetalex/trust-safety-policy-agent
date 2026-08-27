# Trust & Safety Policy Agent

A policy-grounded case adjudication and regression evaluation system. The
project is intentionally structured around case decisions, evidence, Golden Set
evaluation, and bad-case analysis rather than generic document chat.

## Current Status

- Versioned data contract with backward-compatible v1 loading
- Synthetic policy covering five violations and five exemptions
- Golden Set v4 with 210 cases, 60 blind cases, and 90 multimodal fixtures
- Structure-aware Markdown policy parsing with stable chunk IDs
- Offline reproducible embeddings and persistent Chroma retrieval
- Policy-grounded single-case adjudication with evidence and review fallback
- Strict JSON Schema multimodal inference with provider telemetry
- Streamlit workspace with case decisions, Top-K search, and chunk preview
- Evaluation reports, confusion matrices, slices, error buckets, and quality gates
- Frozen Day 4-7 baselines and versioned dataset quota governance

## Project Layout

```text
trust-safety-policy-agent/
├── app.py                         # Streamlit entry point (Day 2+)
├── assets/                        # Screenshots and architecture diagrams
├── data/
│   ├── baselines/                 # Frozen baseline manifests
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
`policy-chroma` Docker volume.

For multimodal inference, copy `.env.example` to `.env`, configure a new API
key and model, then restart:

```bash
docker compose up --build -d
docker compose logs -f policy-agent
```

Use `APP_PORT` to expose a different host port:

```bash
APP_PORT=8080 docker compose up -d
```

Stop the service with `docker compose down`. Add `--volumes` only when the
persisted policy index should also be deleted.

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

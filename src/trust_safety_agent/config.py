"""Project paths and environment-backed runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(
    os.getenv("APP_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data/policies/synthetic_policy_v2.md"
DEFAULT_BRAND_LIBRARY_PATH = PROJECT_ROOT / "data/brands/controlled_brands_v1.csv"
DEFAULT_REVIEW_RECORDS_PATH = PROJECT_ROOT / "data/reviewer/review_records.jsonl"
DEFAULT_STAGING_DATASET_PATH = PROJECT_ROOT / "data/golden_set/staging/candidates.jsonl"
DEFAULT_TRACE_PATH = PROJECT_ROOT / "data/traces/execution_traces.jsonl"
DEFAULT_EVALUATION_RUNS_PATH = PROJECT_ROOT / "data/eval_runs"
DEFAULT_GOLDEN_SET_V4_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"
DEFAULT_GOLDEN_SET_V5_CANDIDATE_PATH = (
    PROJECT_ROOT / "data/golden_set/v5_candidates/golden_set_v5_candidates.jsonl"
)
DEFAULT_DEMO_ASSET_MANIFEST_PATH = (
    PROJECT_ROOT / "data/demo_assets/v5/asset_manifest.json"
)
DEFAULT_SYNTHETIC_DEMO_CASES_PATH = (
    PROJECT_ROOT / "data/golden_set/v5_candidates/golden_set_v5_candidates.jsonl"
)
DEFAULT_PROMPT_PRESETS_PATH = PROJECT_ROOT / "config/prompts_v1.json"
DEFAULT_SKILL_PRESETS_PATH = PROJECT_ROOT / "config/skills_v1.json"
DEFAULT_CUSTOM_PROMPTS_PATH = PROJECT_ROOT / "data/prompt_studio/custom_prompts.jsonl"
DEFAULT_CUSTOM_SKILLS_PATH = PROJECT_ROOT / "data/prompt_studio/custom_skills.jsonl"
DEFAULT_BRAND_KNOWLEDGE_PATH = PROJECT_ROOT / "data/knowledge/brand_product_knowledge_v1.md"
DEFAULT_CASE_STORE_PATH = PROJECT_ROOT / "data/case_store/runtime"
DEFAULT_AGENT_RUNS_PATH = PROJECT_ROOT / "data/runs/agent_runs.jsonl"
DEFAULT_REVIEWER_ANNOTATIONS_PATH = PROJECT_ROOT / "data/runs/reviewer_annotations.jsonl"
DEFAULT_BAD_CASES_PATH = PROJECT_ROOT / "data/bad_cases/bad_cases.jsonl"
DEFAULT_REVIEWER_PROMPTS_PATH = PROJECT_ROOT / "data/reviewer_prompts/prompts.jsonl"
DEFAULT_WORKFLOW_RELEASES_PATH = PROJECT_ROOT / "data/workflow_releases/releases.jsonl"
DEFAULT_KNOWLEDGE_DOCUMENTS_PATH = PROJECT_ROOT / "data/knowledge_uploads"
DEFAULT_MODEL_EXPERIMENTS_PATH = PROJECT_ROOT / "data/model_experiments"
DEFAULT_PHASE9_SHOP_DEMOS_PATH = PROJECT_ROOT / "data/case_store/phase9_demo_shop_cases.jsonl"
DEFAULT_PHASE9_PRODUCT_DEMOS_PATH = PROJECT_ROOT / "data/case_store/phase9_demo_product_cases.jsonl"
DEFAULT_CHROMA_DIRECTORY = Path(
    os.getenv(
        "POLICY_DB_PATH",
        str(PROJECT_ROOT / "data/chroma"),
    )
).expanduser()

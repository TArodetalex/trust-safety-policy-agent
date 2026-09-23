"""Project paths and environment-backed runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(
    os.getenv("APP_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data/policies/mock_policy_v1.md"
DEFAULT_BRAND_LIBRARY_PATH = PROJECT_ROOT / "data/brands/controlled_brands_v1.csv"
DEFAULT_REVIEW_RECORDS_PATH = PROJECT_ROOT / "data/reviewer/review_records.jsonl"
DEFAULT_STAGING_DATASET_PATH = PROJECT_ROOT / "data/golden_set/staging/candidates.jsonl"
DEFAULT_TRACE_PATH = PROJECT_ROOT / "data/traces/execution_traces.jsonl"
DEFAULT_EVALUATION_RUNS_PATH = PROJECT_ROOT / "data/eval_runs"
DEFAULT_GOLDEN_SET_V4_PATH = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"
DEFAULT_PROMPT_PRESETS_PATH = PROJECT_ROOT / "config/prompts_v1.json"
DEFAULT_SKILL_PRESETS_PATH = PROJECT_ROOT / "config/skills_v1.json"
DEFAULT_CUSTOM_PROMPTS_PATH = PROJECT_ROOT / "data/prompt_studio/custom_prompts.jsonl"
DEFAULT_CUSTOM_SKILLS_PATH = PROJECT_ROOT / "data/prompt_studio/custom_skills.jsonl"
DEFAULT_CHROMA_DIRECTORY = Path(
    os.getenv(
        "POLICY_DB_PATH",
        str(PROJECT_ROOT / "data/chroma"),
    )
).expanduser()

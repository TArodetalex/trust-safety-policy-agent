"""Project paths and environment-backed runtime settings."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "data/policies/mock_policy_v1.md"
DEFAULT_CHROMA_DIRECTORY = Path(
    os.getenv(
        "POLICY_DB_PATH",
        str(PROJECT_ROOT / "data/chroma"),
    )
).expanduser()

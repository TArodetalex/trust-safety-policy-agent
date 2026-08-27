"""Validate a Golden Set against its versioned data contract."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from trust_safety_agent.dataset import load_cases, normalize_row, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    try:
        cases = load_cases(path)
    except (OSError, ValidationError, KeyError) as exc:
        print(f"Golden Set validation failed: {exc}", file=sys.stderr)
        return 1

    errors = validate_dataset(cases, project_root=PROJECT_ROOT)
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    decision_counts = Counter(case.human_decision.value for case in cases)
    route_counts = Counter(case.expected_route.value for case in cases)
    print(
        "Golden Set valid: "
        f"{len(cases)} cases, "
        f"{decision_counts['reject']} reject, "
        f"{decision_counts['approve']} approve, "
        f"{sum(case.is_boundary_case for case in cases)} boundary, "
        f"{route_counts['human_review']} expected review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

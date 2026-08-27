"""Verify frozen baseline inputs, fingerprints, and expected metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from trust_safety_agent.baseline import verify_baseline_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data/baselines/day4_rules_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_baseline_manifest(
        args.manifest,
        PROJECT_ROOT,
        require_report=args.require_report,
    )
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.errors:
        print(f"Baseline {result.baseline_id} is invalid:")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print(f"Baseline {result.baseline_id} verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit a Golden Set against the versioned expansion quota plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from trust_safety_agent.dataset import load_cases
from trust_safety_agent.evaluation import fingerprint_cases
from trust_safety_agent.planning import audit_dataset_quotas, load_dataset_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/golden_set/golden_set_v4.csv"
DEFAULT_PLAN = (
    PROJECT_ROOT / "data/golden_set/plans/golden_set_v4_plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_dataset_plan(args.plan)
    cases = load_cases(args.dataset)
    results = audit_dataset_quotas(cases, plan)

    seed_path = PROJECT_ROOT / plan["seed"]["path"]
    if args.dataset.resolve() == seed_path.resolve():
        fingerprint = fingerprint_cases(cases)
        expected = plan["seed"]["normalized_fingerprint"]
        if fingerprint != expected:
            print(
                "Seed fingerprint mismatch: "
                f"expected {expected}, found {fingerprint}"
            )
            return 2

    print(
        f"Dataset quota audit: {len(cases)}/{plan['target_total']} cases "
        f"for {plan['dataset_version']}"
    )
    print("group\tlabel\tcurrent\trequired\tgap")
    for result in results:
        print(
            f"{result.group}\t{result.label}\t{result.current}\t"
            f"{result.required}\t{result.gap}"
        )

    gaps = [result for result in results if not result.met]
    print(
        f"Quota status: {len(results) - len(gaps)}/{len(results)} met; "
        f"{len(gaps)} remaining."
    )
    return 1 if args.strict and gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Day 3 adjudicator against the Golden Set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.schema import ContentType
from trust_safety_agent.vector_store import PolicyVectorStore


DATASET = Path(__file__).parents[1] / "data/golden_set/golden_set_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    agent = PolicyAdjudicator(store)

    with DATASET.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    correct = 0
    review = 0
    for row in rows:
        decision = agent.adjudicate(
            case_id=row["case_id"],
            content_type=ContentType(row["content_type"]),
            input_text=row["input_text"],
        )
        review += decision.decision.value == "need_review"
        label = decision.policy_label.value if decision.policy_label else ""
        matches = decision.decision.value == row["human_decision"] and (
            row["human_decision"] == "approve" or label == row["expected_policy"]
        )
        correct += matches
        if not matches:
            print(
                f"{row['case_id']}: expected "
                f"{row['human_decision']}/{row['expected_policy'] or '-'}, got "
                f"{decision.decision.value}/{label or '-'}"
            )

    accuracy = correct / len(rows) if rows else 0.0
    print(
        f"Golden smoke: {correct}/{len(rows)} matched "
        f"({accuracy:.1%}); need_review={review}."
    )
    return 1 if args.strict and correct != len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

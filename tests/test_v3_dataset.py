from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from trust_safety_agent.dataset import load_cases, validate_dataset
from trust_safety_agent.planning import (
    audit_dataset_quotas,
    load_dataset_plan,
)
from trust_safety_agent.schema import DatasetSplit
from trust_safety_agent.schema import AgentDecisionLabel
from scripts.run_evaluation import build_decider


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/golden_set/golden_set_v3.csv"
PLAN = ROOT / "data/golden_set/plans/golden_set_v3_plan.json"
SPLIT_MANIFEST = ROOT / "data/golden_set/splits/v3/split_manifest.json"


def test_v3_dataset_and_assets_are_valid() -> None:
    cases = load_cases(DATASET)

    assert validate_dataset(cases, project_root=ROOT) == []
    assert len(cases) == 150
    assert sum(bool(case.image_path or case.image_url) for case in cases) == 30


def test_v3_meets_every_planned_quota() -> None:
    results = audit_dataset_quotas(
        load_cases(DATASET),
        load_dataset_plan(PLAN),
    )

    assert results
    assert all(result.met for result in results)


def test_v3_blind_split_is_balanced_and_disjoint() -> None:
    cases = load_cases(DATASET)
    blind = [case for case in cases if case.split == DatasetSplit.BLIND]
    tuning = [case for case in cases if case.split != DatasetSplit.BLIND]

    assert len(blind) == 40
    assert Counter(case.human_decision.value for case in blind) == {
        "reject": 20,
        "approve": 20,
    }
    assert {case.case_id for case in blind}.isdisjoint(
        case.case_id for case in tuning
    )
    assert len({case.case_id for case in blind + tuning}) == 150


def test_v3_case_inputs_are_unique_across_modalities() -> None:
    cases = load_cases(DATASET)
    signatures = {
        (
            case.content_type.value,
            case.input_text.casefold(),
            case.image_path or str(case.image_url or ""),
        )
        for case in cases
    }

    assert len(signatures) == len(cases)


def test_split_files_match_full_dataset() -> None:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    def case_ids(relative_path: str) -> set[str]:
        with (ROOT / relative_path).open(newline="", encoding="utf-8") as source:
            return {row["case_id"] for row in csv.DictReader(source)}

    full = case_ids(manifest["full"]["path"])
    tuning = case_ids(manifest["tuning"]["path"])
    blind = case_ids(manifest["blind"]["path"])

    assert len(full) == manifest["full"]["count"] == 150
    assert len(tuning) == manifest["tuning"]["count"] == 110
    assert len(blind) == manifest["blind"]["count"] == 40
    assert tuning.isdisjoint(blind)
    assert tuning | blind == full


def test_new_cases_have_synthetic_review_ledger_entries() -> None:
    ledger_path = ROOT / "data/golden_set/splits/v3/annotation_ledger_v3.csv"
    with ledger_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == 80
    assert {row["case_id"] for row in rows} == {
        f"C{number:03d}" for number in range(71, 151)
    }
    assert {row["status"] for row in rows} == {"synthetic_verified"}
    assert all("independent human signoff" in row["notes"] for row in rows)


def test_rules_engine_abstains_from_local_image_evidence() -> None:
    image_case = next(
        case for case in load_cases(DATASET) if case.image_path is not None
    )
    decide, version = build_decider("rules", None)  # type: ignore[arg-type]

    decision = decide(image_case)

    assert version == "rules-v1"
    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert "cannot inspect image evidence" in decision.reason

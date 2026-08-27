from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_golden_set_v4 import fixture_specs
from trust_safety_agent.dataset import load_cases, validate_dataset
from trust_safety_agent.planning import audit_dataset_quotas, load_dataset_plan
from trust_safety_agent.schema import DatasetSplit


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/golden_set/golden_set_v4.csv"
PLAN = ROOT / "data/golden_set/plans/golden_set_v4_plan.json"
SPLIT_MANIFEST = ROOT / "data/golden_set/splits/v4/split_manifest.json"
ASSET_MANIFEST = ROOT / "data/golden_set/assets/v4/asset_manifest.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_dataset_and_assets_are_valid() -> None:
    cases = load_cases(DATASET)

    assert validate_dataset(cases, project_root=ROOT) == []
    assert len(cases) == 210
    assert sum(bool(case.image_path or case.image_url) for case in cases) == 90


def test_v4_meets_every_planned_quota() -> None:
    results = audit_dataset_quotas(
        load_cases(DATASET),
        load_dataset_plan(PLAN),
    )

    assert results
    assert all(result.met for result in results)


def test_v4_new_fixture_design_is_balanced() -> None:
    cases = [
        case for case in load_cases(DATASET)
        if int(case.case_id[1:]) >= 151
    ]

    assert len(fixture_specs()) == len(cases) == 60
    assert Counter(case.human_decision.value for case in cases) == {
        "reject": 30,
        "approve": 30,
    }
    assert Counter(case.split for case in cases) == {
        DatasetSplit.REGRESSION: 24,
        DatasetSplit.DEVELOPMENT: 16,
        DatasetSplit.BLIND: 20,
    }
    assert sum("clean_generic" in case.tags for case in cases) == 10
    assert sum(case.expected_route.value == "human_review" for case in cases) == 20


def test_v4_split_and_asset_manifests_match_files() -> None:
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    for name in ("full", "tuning", "blind"):
        item = split_manifest[name]
        path = ROOT / item["path"]
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == item["count"]
        assert sha256(path) == item["sha256"]

    with ASSET_MANIFEST.open(newline="", encoding="utf-8") as source:
        assets = list(csv.DictReader(source))
    assert len(assets) == 60
    for asset in assets:
        path = ROOT / asset["relative_path"]
        assert path.is_file()
        assert sha256(path) == asset["sha256"]


def test_v4_annotation_ledger_covers_every_new_case() -> None:
    ledger = ROOT / "data/golden_set/splits/v4/annotation_ledger_v4.csv"
    with ledger.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    assert {row["case_id"] for row in rows} == {
        f"C{number:03d}" for number in range(151, 211)
    }
    assert {row["status"] for row in rows} == {"synthetic_verified"}
    assert all("independent human signoff" in row["notes"] for row in rows)

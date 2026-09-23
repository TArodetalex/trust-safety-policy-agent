from __future__ import annotations

import json
from pathlib import Path

from trust_safety_agent.baseline import (
    sha256_file,
    verify_baseline_manifest,
)
from trust_safety_agent.dataset import load_cases
from trust_safety_agent.evaluation import EvaluationGates
from trust_safety_agent.planning import (
    audit_dataset_quotas,
    load_dataset_plan,
)
from trust_safety_agent.schema import DatasetSplit


ROOT = Path(__file__).parents[1]
GATES_PATH = ROOT / "config/evaluation_gates_v1.json"
MANIFEST_PATH = ROOT / "data/baselines/day4_rules_v1.json"
OCR_MANIFEST_PATH = ROOT / "data/baselines/day5_rules_ocr_v1.json"
PRODUCTION_MANIFEST_PATH = (
    ROOT / "data/baselines/day6_production_offline_v1.json"
)
DAY7_MANIFEST_PATH = (
    ROOT / "data/baselines/day7_production_offline_v1.json"
)
DATASET_PATH = ROOT / "data/golden_set/golden_set_v2.csv"
PLAN_PATH = ROOT / "data/golden_set/plans/golden_set_v3_plan.json"


def test_frozen_gates_load_from_versioned_config() -> None:
    gates = EvaluationGates.from_json(GATES_PATH)

    assert gates == EvaluationGates()
    assert gates.as_dict()["min_auto_accuracy"] == 0.95


def test_day4_baseline_manifest_verifies() -> None:
    result = verify_baseline_manifest(MANIFEST_PATH, ROOT)

    assert result.valid
    assert result.baseline_id == "EV-DAY4-RULES-V1"


def test_day5_ocr_baseline_manifest_verifies() -> None:
    result = verify_baseline_manifest(OCR_MANIFEST_PATH, ROOT)

    assert result.valid
    assert result.baseline_id == "EV-DAY5-RULES-OCR-V1"


def test_day6_production_baseline_manifest_verifies() -> None:
    result = verify_baseline_manifest(PRODUCTION_MANIFEST_PATH, ROOT)

    assert result.valid
    assert result.baseline_id == "EV-DAY6-PRODUCTION-OFFLINE-V1"


def test_day7_production_baseline_manifest_verifies() -> None:
    result = verify_baseline_manifest(DAY7_MANIFEST_PATH, ROOT)

    assert result.valid
    assert result.baseline_id == "EV-DAY7-PRODUCTION-OFFLINE-V1"
    assert all("source drift" in warning for warning in result.warnings)


def test_baseline_verification_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "dataset.csv"
    artifact.write_text("original", encoding="utf-8")
    policy = tmp_path / "policy.md"
    policy.write_text("policy", encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text("{}", encoding="utf-8")
    manifest = {
        "baseline_id": "EV-TAMPER",
        "dataset": {
            "path": "dataset.csv",
            "sha256": sha256_file(artifact),
            "normalized_fingerprint": "not-evaluated",
        },
        "policy": {"path": "policy.md", "sha256": sha256_file(policy)},
        "gates": {"path": "gates.json", "sha256": sha256_file(gates)},
        "expected_metrics": {},
        "report_path": "missing-report.json",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact.write_text("modified", encoding="utf-8")

    result = verify_baseline_manifest(manifest_path, tmp_path)

    assert not result.valid
    assert any("dataset sha256 mismatch" in error for error in result.errors)


def test_text_artifact_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    lf_file = tmp_path / "lf.json"
    crlf_file = tmp_path / "crlf.json"
    lf_file.write_bytes(b'{\n  "version": 1\n}\n')
    crlf_file.write_bytes(b'{\r\n  "version": 1\r\n}\r\n')

    assert sha256_file(lf_file) == sha256_file(crlf_file)


def test_v3_plan_exposes_seed_dataset_gaps() -> None:
    cases = load_cases(DATASET_PATH)
    plan = load_dataset_plan(PLAN_PATH)
    results = {
        (result.group, result.label): result
        for result in audit_dataset_quotas(cases, plan)
    }

    assert results[("dataset", "total")].gap == 80
    assert results[("dataset", "image_cases")].gap == 30
    assert results[("splits", "blind")].gap == 40
    assert results[("policies", "counterfeit")].gap == 6
    assert results[("tags", "multimodal")].gap == 30


def test_blind_split_is_part_of_the_contract() -> None:
    assert DatasetSplit("blind") == DatasetSplit.BLIND

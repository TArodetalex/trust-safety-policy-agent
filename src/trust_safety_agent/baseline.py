"""Verification helpers for frozen evaluation baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from trust_safety_agent.dataset import load_cases
from trust_safety_agent.evaluation import fingerprint_cases


@dataclass
class BaselineVerification:
    baseline_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_baseline_manifest(
    manifest_path: Path,
    project_root: Path,
    require_report: bool = False,
) -> BaselineVerification:
    manifest = load_baseline_manifest(manifest_path)
    result = BaselineVerification(baseline_id=manifest["baseline_id"])

    for section_name in ("dataset", "policy", "gates"):
        section = manifest[section_name]
        artifact_path = project_root / section["path"]
        if not artifact_path.is_file():
            result.errors.append(
                f"{section_name} artifact is missing: {section['path']}"
            )
            continue
        actual_hash = sha256_file(artifact_path)
        if actual_hash != section["sha256"]:
            result.errors.append(
                f"{section_name} sha256 mismatch: "
                f"expected {section['sha256']}, found {actual_hash}"
            )

    for artifact in manifest.get("artifacts", []):
        artifact_path = project_root / artifact["path"]
        if not artifact_path.is_file():
            result.errors.append(
                f"baseline artifact is missing: {artifact['path']}"
            )
            continue
        actual_hash = sha256_file(artifact_path)
        if actual_hash != artifact["sha256"]:
            result.errors.append(
                f"artifact sha256 mismatch for {artifact['path']}: "
                f"expected {artifact['sha256']}, found {actual_hash}"
            )

    dataset_path = project_root / manifest["dataset"]["path"]
    if dataset_path.is_file():
        try:
            actual_fingerprint = fingerprint_cases(load_cases(dataset_path))
        except Exception as exc:
            result.errors.append(
                "dataset could not be normalized: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            expected_fingerprint = manifest["dataset"][
                "normalized_fingerprint"
            ]
            if actual_fingerprint != expected_fingerprint:
                result.errors.append(
                    "dataset normalized fingerprint mismatch: "
                    f"expected {expected_fingerprint}, "
                    f"found {actual_fingerprint}"
                )

    report_path = project_root / manifest["report_path"]
    if not report_path.is_file():
        message = f"baseline report is missing: {manifest['report_path']}"
        if require_report:
            result.errors.append(message)
        else:
            result.warnings.append(message)
        return result

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["eval_run_id"] != manifest["baseline_id"]:
        result.errors.append("report eval_run_id does not match baseline_id")
    if (
        report["dataset_fingerprint"]
        != manifest["dataset"]["normalized_fingerprint"]
    ):
        result.errors.append("report dataset fingerprint does not match manifest")

    for metric, expected in manifest["expected_metrics"].items():
        actual = report["metrics"].get(metric)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                actual,
                expected,
                rel_tol=0,
                abs_tol=1e-12,
            )
        else:
            matches = actual == expected
        if not matches:
            result.errors.append(
                f"metric {metric} mismatch: expected {expected}, found {actual}"
            )
    return result

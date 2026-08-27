"""Golden Set loading and dataset-level validation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from trust_safety_agent.schema import (
    ExemptionType,
    ExpectedRoute,
    GoldenCase,
    PolicyLabel,
)


def normalize_row(row: Dict[str, Optional[str]]) -> Dict[str, object]:
    normalized: Dict[str, object] = dict(row)
    normalized["image_url"] = row.get("image_url") or None
    normalized["image_path"] = row.get("image_path") or None
    normalized["expected_policy"] = row.get("expected_policy") or None
    normalized["is_boundary_case"] = (
        (row.get("is_boundary_case") or "").lower() == "true"
    )
    normalized["expected_route"] = (
        row.get("expected_route") or ExpectedRoute.AUTO_DECIDE.value
    )
    normalized["tags"] = [
        tag.strip()
        for tag in (row.get("tags") or "").split("|")
        if tag.strip()
    ]
    return normalized


def load_cases(path: Path) -> List[GoldenCase]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return [
            GoldenCase.model_validate(normalize_row(row))
            for row in reader
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_multimodal_assets(
    cases: List[GoldenCase],
    project_root: Path,
) -> List[str]:
    errors: List[str] = []
    image_paths = {
        case.image_path
        for case in cases
        if case.image_path is not None
    }
    manifest_directories = {
        str(PurePosixPath(relative_path).parent)
        for relative_path in image_paths
    }
    manifest: Dict[str, Dict[str, str]] = {}

    for relative_directory in sorted(manifest_directories):
        manifest_path = project_root / relative_directory / "asset_manifest.csv"
        if not manifest_path.is_file():
            errors.append(
                f"multimodal asset manifest is missing: {manifest_path}"
            )
            continue
        with manifest_path.open(newline="", encoding="utf-8") as source:
            manifest_rows = list(csv.DictReader(source))
        directory_manifest = {
            row["relative_path"]: row
            for row in manifest_rows
        }
        if len(directory_manifest) != len(manifest_rows):
            errors.append(
                "asset manifest relative_path values must be unique: "
                f"{manifest_path}"
            )
        expected_paths = {
            path
            for path in image_paths
            if str(PurePosixPath(path).parent) == relative_directory
        }
        if set(directory_manifest) != expected_paths:
            errors.append(
                "asset manifest paths must exactly match dataset image paths "
                f"for {relative_directory}"
            )
        manifest.update(directory_manifest)

    for relative_path in sorted(image_paths):
        asset_path = project_root / relative_path
        if not asset_path.is_file():
            errors.append(f"multimodal asset is missing: {relative_path}")
            continue
        row = manifest.get(relative_path)
        if row and _sha256(asset_path) != row["sha256"]:
            errors.append(f"multimodal asset sha256 mismatch: {relative_path}")
    return errors


def validate_dataset(
    cases: List[GoldenCase],
    project_root: Optional[Path] = None,
) -> List[str]:
    errors: List[str] = []
    if not cases:
        return ["dataset must contain at least one case"]

    case_ids = [case.case_id for case in cases]
    versions = {case.dataset_version for case in cases}
    schema_versions = {case.schema_version for case in cases}
    decision_counts = Counter(case.human_decision.value for case in cases)
    route_counts = Counter(case.expected_route.value for case in cases)
    split_counts = Counter(case.split.value for case in cases)
    policies = {case.expected_policy for case in cases if case.expected_policy}
    exemptions = {
        case.expected_exemption
        for case in cases
        if case.expected_exemption != ExemptionType.NONE
    }
    tags = {tag for case in cases for tag in case.tags}
    boundary_count = sum(case.is_boundary_case for case in cases)
    image_count = sum(
        case.image_url is not None or case.image_path is not None
        for case in cases
    )

    if len(case_ids) != len(set(case_ids)):
        errors.append("case_id values must be unique")
    if len(versions) != 1:
        errors.append(f"dataset_version must be uniform, found {sorted(versions)}")
        return errors
    if policies != set(PolicyLabel):
        errors.append("all five violation policies must be represented")
    if exemptions != set(ExemptionType) - {ExemptionType.NONE}:
        errors.append("all five exemptions must be represented")

    version = next(iter(versions))
    if version == "v1.0.0":
        if schema_versions != {"1.0.0"}:
            errors.append("dataset v1.0.0 requires schema_version 1.0.0")
        if len(cases) != 50:
            errors.append(f"expected 50 cases, found {len(cases)}")
        if decision_counts != {"reject": 25, "approve": 25}:
            errors.append(
                f"expected balanced decisions, found {dict(decision_counts)}"
            )
        if boundary_count < 20:
            errors.append(
                f"expected at least 20 boundary cases, found {boundary_count}"
            )
        return errors

    if version == "v2.0.0":
        if schema_versions != {"1.1.0"}:
            errors.append("dataset v2.0.0 requires schema_version 1.1.0")
        if len(cases) != 70:
            errors.append(f"expected 70 cases, found {len(cases)}")
        if min(decision_counts.values(), default=0) < 30:
            errors.append(
                "v2 requires at least 30 approve and 30 reject cases"
            )
        if route_counts[ExpectedRoute.HUMAN_REVIEW.value] < 10:
            errors.append("v2 requires at least 10 human-review routing cases")
        if boundary_count < 30:
            errors.append(
                f"expected at least 30 boundary cases, found {boundary_count}"
            )
        required_tags = {
            "brand_only",
            "conflicting_signals",
            "counterfeit",
            "exemption",
            "insufficient_context",
            "unbranded",
        }
        missing_tags = sorted(required_tags - tags)
        if missing_tags:
            errors.append(f"v2 is missing required tags: {missing_tags}")
        return errors

    if version == "v3.0.0":
        if schema_versions != {"1.2.0"}:
            errors.append("dataset v3.0.0 requires schema_version 1.2.0")
        if len(cases) != 150:
            errors.append(f"expected 150 cases, found {len(cases)}")
        expected_ids = {f"C{number:03d}" for number in range(1, 151)}
        if set(case_ids) != expected_ids:
            errors.append("v3 requires the contiguous case range C001-C150")
        if min(decision_counts.values(), default=0) < 70:
            errors.append("v3 requires at least 70 approve and 70 reject cases")
        expected_splits = {
            "regression": 80,
            "development": 30,
            "blind": 40,
        }
        if split_counts != expected_splits:
            errors.append(
                f"v3 split counts must be {expected_splits}, "
                f"found {dict(split_counts)}"
            )
        if image_count != 30:
            errors.append(
                f"v3 requires exactly 30 multimodal cases, found {image_count}"
            )
        if route_counts[ExpectedRoute.HUMAN_REVIEW.value] < 25:
            errors.append("v3 requires at least 25 human-review routing cases")
        if boundary_count < 90:
            errors.append(
                f"v3 requires at least 90 boundary cases, found {boundary_count}"
            )
        policy_counts = Counter(
            case.expected_policy.value
            for case in cases
            if case.expected_policy
        )
        exemption_counts = Counter(
            case.expected_exemption.value
            for case in cases
            if case.expected_exemption != ExemptionType.NONE
        )
        for policy in PolicyLabel:
            if policy_counts[policy.value] < 15:
                errors.append(f"v3 requires at least 15 {policy.value} cases")
        for exemption in set(ExemptionType) - {ExemptionType.NONE}:
            if exemption_counts[exemption.value] < 10:
                errors.append(
                    f"v3 requires at least 10 {exemption.value} exemptions"
                )
        required_tags = {
            "brand_only",
            "conflicting_signals",
            "explicit_signal",
            "exemption",
            "insufficient_context",
            "multimodal",
        }
        missing_tags = sorted(required_tags - tags)
        if missing_tags:
            errors.append(f"v3 is missing required tags: {missing_tags}")
        if project_root is not None:
            errors.extend(validate_multimodal_assets(cases, project_root))
        return errors

    if version == "v4.0.0":
        if schema_versions != {"1.2.0"}:
            errors.append("dataset v4.0.0 requires schema_version 1.2.0")
        if len(cases) != 210:
            errors.append(f"expected 210 cases, found {len(cases)}")
        expected_ids = {f"C{number:03d}" for number in range(1, 211)}
        if set(case_ids) != expected_ids:
            errors.append("v4 requires the contiguous case range C001-C210")
        if decision_counts != {"reject": 105, "approve": 105}:
            errors.append(
                "v4 requires 105 approve and 105 reject cases, "
                f"found {dict(decision_counts)}"
            )
        expected_splits = {
            "regression": 104,
            "development": 46,
            "blind": 60,
        }
        if split_counts != expected_splits:
            errors.append(
                f"v4 split counts must be {expected_splits}, "
                f"found {dict(split_counts)}"
            )
        if image_count != 90:
            errors.append(
                f"v4 requires exactly 90 multimodal cases, found {image_count}"
            )
        if route_counts[ExpectedRoute.HUMAN_REVIEW.value] != 45:
            errors.append(
                "v4 requires exactly 45 human-review routing cases"
            )
        if boundary_count < 180:
            errors.append(
                f"v4 requires at least 180 boundary cases, found {boundary_count}"
            )
        day7_cases = [case for case in cases if "day7" in case.tags]
        if len(day7_cases) != 60:
            errors.append(
                f"v4 requires exactly 60 Day 7 cases, found {len(day7_cases)}"
            )
        if any("shadow" not in case.tags for case in day7_cases):
            errors.append("all Day 7 cases must be tagged for shadow evaluation")
        if sum("clean_generic" in case.tags for case in day7_cases) != 10:
            errors.append("v4 requires exactly 10 clean generic controls")
        if sum(
            case.split.value == "blind" and "day7" in case.tags
            for case in cases
        ) != 20:
            errors.append("v4 requires exactly 20 new blind visual cases")
        if project_root is not None:
            errors.extend(validate_multimodal_assets(cases, project_root))
        return errors

    errors.append(f"unsupported dataset_version: {version}")
    return errors

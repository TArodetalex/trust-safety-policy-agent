"""Quota planning for Golden Set expansion."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from trust_safety_agent.schema import ExemptionType, GoldenCase


@dataclass(frozen=True)
class QuotaResult:
    group: str
    label: str
    current: int
    required: int

    @property
    def gap(self) -> int:
        return max(0, self.required - self.current)

    @property
    def met(self) -> bool:
        return self.gap == 0


def load_dataset_plan(path: Path) -> Dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["target_total"] < plan["seed"]["case_count"]:
        raise ValueError("target_total cannot be smaller than the seed dataset")
    return plan


def _append_mapping_quotas(
    results: List[QuotaResult],
    group: str,
    counts: Counter[str],
    required: Dict[str, int],
) -> None:
    for label, minimum in required.items():
        results.append(
            QuotaResult(
                group=group,
                label=label,
                current=counts[label],
                required=minimum,
            )
        )


def audit_dataset_quotas(
    cases: Sequence[GoldenCase],
    plan: Dict[str, Any],
) -> List[QuotaResult]:
    minimums = plan["minimums"]
    results = [
        QuotaResult("dataset", "total", len(cases), plan["target_total"]),
        QuotaResult(
            "dataset",
            "boundary_cases",
            sum(case.is_boundary_case for case in cases),
            minimums["boundary_cases"],
        ),
        QuotaResult(
            "dataset",
            "image_cases",
            sum(
                case.image_url is not None or case.image_path is not None
                for case in cases
            ),
            minimums["image_cases"],
        ),
        QuotaResult(
            "dataset",
            "human_review_cases",
            sum(case.expected_route.value == "human_review" for case in cases),
            minimums["human_review_cases"],
        ),
    ]

    decision_counts = Counter(case.human_decision.value for case in cases)
    split_counts = Counter(case.split.value for case in cases)
    content_counts = Counter(case.content_type.value for case in cases)
    risk_counts = Counter(case.risk_level.value for case in cases)
    policy_counts = Counter(
        case.expected_policy.value
        for case in cases
        if case.expected_policy is not None
    )
    exemption_counts = Counter(
        case.expected_exemption.value
        for case in cases
        if case.expected_exemption != ExemptionType.NONE
    )
    tag_counts = Counter(tag for case in cases for tag in case.tags)

    quota_groups = [
        ("decisions", decision_counts),
        ("splits", split_counts),
        ("content_types", content_counts),
        ("risk_levels", risk_counts),
        ("policies", policy_counts),
        ("exemptions", exemption_counts),
        ("tags", tag_counts),
    ]
    for group, counts in quota_groups:
        _append_mapping_quotas(
            results,
            group,
            counts,
            minimums[group],
        )
    return results

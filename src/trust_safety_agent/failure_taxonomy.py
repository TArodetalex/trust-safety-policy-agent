"""Deterministic failure classification and evaluation aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from pydantic import Field

from trust_safety_agent.schema import (
    ErrorType,
    EvaluationRecord,
    FailureType,
    StrictModel,
)


class FailureAggregation(StrictModel):
    total_error_cases: int = Field(ge=0)
    failure_counts: Dict[str, int] = Field(default_factory=dict)
    by_error_type: Dict[str, Dict[str, int]] = Field(default_factory=dict)


def classify_evaluation_failure(record: EvaluationRecord) -> FailureType:
    if record.error_type == ErrorType.NONE:
        return FailureType.NONE
    if record.error_type == ErrorType.INVALID_OUTPUT:
        return FailureType.SCHEMA_ERROR
    if record.error_type == ErrorType.WRONG_POLICY:
        return FailureType.POLICY_MISREAD
    if record.error_type in {
        ErrorType.UNEXPECTED_AUTO_DECISION,
        ErrorType.UNSAFE_AUTO_DECISION,
    }:
        return FailureType.ROUTING_ERROR
    if record.error_type == ErrorType.UNEXPECTED_REVIEW:
        reason = record.agent_reason.casefold()
        if any(term in reason for term in ("image", "ocr", "visual", "图片", "视觉")):
            return FailureType.EVIDENCE_MISS
        if any(term in reason for term in ("retriev", "policy evidence", "检索", "策略证据")):
            return FailureType.RETRIEVAL_MISS
        return FailureType.LOW_CONFIDENCE
    if record.error_type == ErrorType.FALSE_REJECT:
        return FailureType.EXEMPTION_MISS
    if record.error_type == ErrorType.FALSE_APPROVE:
        if not record.evidence_chunk_ids:
            return FailureType.EVIDENCE_MISS
        return FailureType.POLICY_MISREAD
    return FailureType.UNKNOWN


def aggregate_failures(records: Sequence[EvaluationRecord]) -> FailureAggregation:
    failures = [
        (
            record,
            record.failure_type
            if record.failure_type != FailureType.NONE
            else classify_evaluation_failure(record),
        )
        for record in records
        if record.error_type != ErrorType.NONE
    ]
    counts = Counter(failure.value for _, failure in failures)
    by_error: Dict[str, Counter[str]] = defaultdict(Counter)
    for record, failure in failures:
        by_error[record.error_type.value][failure.value] += 1
    return FailureAggregation(
        total_error_cases=len(failures),
        failure_counts=dict(sorted(counts.items())),
        by_error_type={
            error_type: dict(sorted(values.items()))
            for error_type, values in sorted(by_error.items())
        },
    )

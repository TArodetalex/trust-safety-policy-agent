"""Selective-classification evaluation for policy adjudicators."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ErrorType,
    EvaluationMetrics,
    EvaluationRecord,
    EvaluationReport,
    ExpectedRoute,
    GoldenCase,
    HumanDecision,
)


DecisionFunction = Callable[[GoldenCase], AgentDecision]


@dataclass(frozen=True)
class EvaluationGates:
    min_auto_accuracy: float = 0.95
    min_route_accuracy: float = 0.90
    min_coverage: float = 0.80
    min_policy_accuracy: float = 0.95
    max_false_approve_rate: float = 0.02
    max_false_reject_rate: float = 0.02

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationGates":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload["thresholds"])

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in self.__dict__.items()
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _case_is_strictly_correct(record: EvaluationRecord) -> bool:
    return record.is_decision_correct and record.is_policy_correct is not False


def _classify_error(
    case: GoldenCase,
    decision: AgentDecision,
    decision_correct: bool,
    policy_correct: Optional[bool],
) -> ErrorType:
    is_review = decision.decision == AgentDecisionLabel.NEED_REVIEW
    expects_review = case.expected_route == ExpectedRoute.HUMAN_REVIEW

    if expects_review:
        if is_review:
            return ErrorType.NONE
        if decision_correct and policy_correct is not False:
            return ErrorType.UNEXPECTED_AUTO_DECISION
        return ErrorType.UNSAFE_AUTO_DECISION

    if is_review:
        return ErrorType.UNEXPECTED_REVIEW
    if (
        case.human_decision == HumanDecision.APPROVE
        and decision.decision == AgentDecisionLabel.REJECT
    ):
        return ErrorType.FALSE_REJECT
    if (
        case.human_decision == HumanDecision.REJECT
        and decision.decision == AgentDecisionLabel.APPROVE
    ):
        return ErrorType.FALSE_APPROVE
    if decision_correct and policy_correct is False:
        return ErrorType.WRONG_POLICY
    return ErrorType.NONE


def _record_for_decision(
    case: GoldenCase,
    decision: AgentDecision,
    eval_run_id: str,
    prompt_version: str,
    retrieval_version: str,
    latency_ms: int,
) -> EvaluationRecord:
    decision_correct = decision.decision.value == case.human_decision.value
    policy_correct = None
    if case.human_decision == HumanDecision.REJECT:
        policy_correct = decision.policy_label == case.expected_policy

    predicted_review = decision.decision == AgentDecisionLabel.NEED_REVIEW
    expected_review = case.expected_route == ExpectedRoute.HUMAN_REVIEW
    route_correct = predicted_review == expected_review
    error_type = _classify_error(
        case,
        decision,
        decision_correct,
        policy_correct,
    )
    return EvaluationRecord(
        eval_run_id=eval_run_id,
        case_id=case.case_id,
        prompt_version=prompt_version,
        retrieval_version=retrieval_version,
        human_decision=case.human_decision,
        expected_policy=case.expected_policy,
        agent_decision=decision.decision,
        agent_policy=decision.policy_label,
        is_decision_correct=decision_correct,
        is_policy_correct=policy_correct,
        expected_route=case.expected_route,
        is_route_correct=route_correct,
        confidence=decision.confidence,
        content_type=case.content_type,
        risk_level=case.risk_level,
        split=case.split,
        is_boundary_case=case.is_boundary_case,
        tags=case.tags,
        agent_reason=decision.reason,
        evidence_chunk_ids=[
            evidence.chunk_id for evidence in decision.matched_policy
        ],
        error_type=error_type,
        latency_ms=latency_ms,
    )


def _invalid_record(
    case: GoldenCase,
    eval_run_id: str,
    prompt_version: str,
    retrieval_version: str,
    latency_ms: int,
    error: Exception,
) -> EvaluationRecord:
    return EvaluationRecord(
        eval_run_id=eval_run_id,
        case_id=case.case_id,
        prompt_version=prompt_version,
        retrieval_version=retrieval_version,
        human_decision=case.human_decision,
        expected_policy=case.expected_policy,
        agent_decision=AgentDecisionLabel.NEED_REVIEW,
        is_decision_correct=False,
        is_policy_correct=(
            False if case.human_decision == HumanDecision.REJECT else None
        ),
        expected_route=case.expected_route,
        is_route_correct=case.expected_route == ExpectedRoute.HUMAN_REVIEW,
        content_type=case.content_type,
        risk_level=case.risk_level,
        split=case.split,
        is_boundary_case=case.is_boundary_case,
        tags=case.tags,
        agent_reason=f"{type(error).__name__}: {error}"[:2000],
        error_type=ErrorType.INVALID_OUTPUT,
        latency_ms=latency_ms,
    )


def calculate_metrics(
    records: Sequence[EvaluationRecord],
    eval_run_id: str,
) -> EvaluationMetrics:
    total = len(records)
    auto_records = [
        record
        for record in records
        if record.agent_decision != AgentDecisionLabel.NEED_REVIEW
    ]
    review_count = total - len(auto_records)
    strict_correct = sum(_case_is_strictly_correct(record) for record in records)
    auto_correct = sum(
        _case_is_strictly_correct(record) for record in auto_records
    )
    route_correct = sum(record.is_route_correct for record in records)

    human_rejects = [
        record
        for record in records
        if record.human_decision == HumanDecision.REJECT
    ]
    human_approves = [
        record
        for record in records
        if record.human_decision == HumanDecision.APPROVE
    ]
    true_rejects = sum(
        record.agent_decision == AgentDecisionLabel.REJECT
        for record in human_rejects
    )
    false_rejects = sum(
        record.agent_decision == AgentDecisionLabel.REJECT
        for record in human_approves
    )
    false_approves = sum(
        record.agent_decision == AgentDecisionLabel.APPROVE
        for record in human_rejects
    )
    auto_rejected_human_rejects = [
        record
        for record in human_rejects
        if record.agent_decision == AgentDecisionLabel.REJECT
    ]
    policy_correct = sum(
        record.is_policy_correct is True
        for record in auto_rejected_human_rejects
    )

    precision = _safe_ratio(true_rejects, true_rejects + false_rejects)
    recall = _safe_ratio(true_rejects, len(human_rejects))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    labels = [
        AgentDecisionLabel.APPROVE.value,
        AgentDecisionLabel.REJECT.value,
        AgentDecisionLabel.NEED_REVIEW.value,
    ]
    confusion: Dict[str, Dict[str, int]] = {
        truth.value: {label: 0 for label in labels}
        for truth in HumanDecision
    }
    for record in records:
        confusion[record.human_decision.value][record.agent_decision.value] += 1

    errors = Counter(
        record.error_type.value
        for record in records
        if record.error_type != ErrorType.NONE
    )
    return EvaluationMetrics(
        eval_run_id=eval_run_id,
        total_cases=total,
        auto_decided_cases=len(auto_records),
        need_review_cases=review_count,
        accuracy=_safe_ratio(strict_correct, total),
        auto_accuracy=_safe_ratio(auto_correct, len(auto_records)),
        coverage=_safe_ratio(len(auto_records), total),
        route_accuracy=_safe_ratio(route_correct, total),
        policy_accuracy=_safe_ratio(
            policy_correct,
            len(auto_rejected_human_rejects),
        ),
        precision=precision,
        recall=recall,
        f1=f1,
        false_reject_rate=_safe_ratio(false_rejects, len(human_approves)),
        false_approve_rate=_safe_ratio(false_approves, len(human_rejects)),
        review_rate=_safe_ratio(review_count, total),
        confusion_matrix=confusion,
        error_counts=dict(sorted(errors.items())),
    )


def _slice_records(
    records: Sequence[EvaluationRecord],
) -> Dict[str, List[EvaluationRecord]]:
    slices: Dict[str, List[EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.content_type:
            slices[f"content_type:{record.content_type.value}"].append(record)
        if record.risk_level:
            slices[f"risk:{record.risk_level.value}"].append(record)
        if record.split:
            slices[f"split:{record.split.value}"].append(record)
        boundary = "boundary" if record.is_boundary_case else "non_boundary"
        slices[f"boundary:{boundary}"].append(record)
        slices[f"route:{record.expected_route.value}"].append(record)
        for tag in record.tags:
            slices[f"tag:{tag}"].append(record)
    return dict(sorted(slices.items()))


def fingerprint_cases(cases: Sequence[GoldenCase]) -> str:
    serialized_cases = "\n".join(
        case.model_dump_json(
            exclude={"image_path"} if case.image_path is None else None
        )
        for case in cases
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(serialized_cases).hexdigest()


def evaluate_cases(
    cases: Sequence[GoldenCase],
    decide: DecisionFunction,
    engine: str,
    prompt_version: str,
    retrieval_version: str,
    gates: Optional[EvaluationGates] = None,
    eval_run_id: Optional[str] = None,
) -> Tuple[EvaluationReport, List[EvaluationRecord]]:
    run_id = eval_run_id or (
        datetime.now(timezone.utc).strftime("EV-%Y%m%dT%H%M%SZ-")
        + uuid4().hex[:8]
    )
    records: List[EvaluationRecord] = []
    for case in cases:
        started = time.perf_counter()
        try:
            decision = decide(case)
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            record = _record_for_decision(
                case,
                decision,
                run_id,
                prompt_version,
                retrieval_version,
                latency_ms,
            )
        except Exception as exc:
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            record = _invalid_record(
                case,
                run_id,
                prompt_version,
                retrieval_version,
                latency_ms,
                exc,
            )
        records.append(record)

    metrics = calculate_metrics(records, run_id)
    slices = {
        name: calculate_metrics(slice_records, f"{run_id}:{name}")
        for name, slice_records in _slice_records(records).items()
    }
    gate_config = gates or EvaluationGates()
    failed_gates = check_gates(metrics, gate_config)
    report = EvaluationReport(
        eval_run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        dataset_version=cases[0].dataset_version if cases else "unknown",
        dataset_fingerprint=fingerprint_cases(cases),
        engine=engine,
        prompt_version=prompt_version,
        retrieval_version=retrieval_version,
        metrics=metrics,
        slices=slices,
        gates=gate_config.as_dict(),
        failed_gates=failed_gates,
    )
    return report, records


def check_gates(
    metrics: EvaluationMetrics,
    gates: EvaluationGates,
) -> List[str]:
    failures = []
    comparisons = [
        (
            "auto_accuracy",
            metrics.auto_accuracy,
            ">=",
            gates.min_auto_accuracy,
        ),
        (
            "route_accuracy",
            metrics.route_accuracy,
            ">=",
            gates.min_route_accuracy,
        ),
        ("coverage", metrics.coverage, ">=", gates.min_coverage),
        (
            "policy_accuracy",
            metrics.policy_accuracy,
            ">=",
            gates.min_policy_accuracy,
        ),
        (
            "false_approve_rate",
            metrics.false_approve_rate,
            "<=",
            gates.max_false_approve_rate,
        ),
        (
            "false_reject_rate",
            metrics.false_reject_rate,
            "<=",
            gates.max_false_reject_rate,
        ),
    ]
    for name, actual, operator, threshold in comparisons:
        failed = actual < threshold if operator == ">=" else actual > threshold
        if failed:
            failures.append(
                f"{name}={actual:.4f} must be {operator} {threshold:.4f}"
            )
    return failures


def write_evaluation_artifacts(
    report: EvaluationReport,
    records: Iterable[EvaluationRecord],
    output_directory: Path,
) -> Dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "report.json"
    records_path = output_directory / "records.jsonl"
    errors_path = output_directory / "errors.csv"
    record_list = list(records)

    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    records_path.write_text(
        "\n".join(record.model_dump_json() for record in record_list) + "\n",
        encoding="utf-8",
    )
    error_rows = [
        record
        for record in record_list
        if record.error_type != ErrorType.NONE
    ]
    with errors_path.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "case_id",
            "error_type",
            "human_decision",
            "expected_policy",
            "agent_decision",
            "agent_policy",
            "expected_route",
            "confidence",
            "latency_ms",
            "tags",
            "agent_reason",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in error_rows:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "error_type": record.error_type.value,
                    "human_decision": record.human_decision.value,
                    "expected_policy": (
                        record.expected_policy.value
                        if record.expected_policy
                        else ""
                    ),
                    "agent_decision": record.agent_decision.value,
                    "agent_policy": (
                        record.agent_policy.value if record.agent_policy else ""
                    ),
                    "expected_route": record.expected_route.value,
                    "confidence": f"{record.confidence:.4f}",
                    "latency_ms": record.latency_ms,
                    "tags": "|".join(record.tags),
                    "agent_reason": record.agent_reason,
                }
            )
    return {
        "report": report_path,
        "records": records_path,
        "errors": errors_path,
    }

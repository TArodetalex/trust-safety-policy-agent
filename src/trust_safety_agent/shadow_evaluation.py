"""Shadow multimodal evaluation records and summaries."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from trust_safety_agent.llm_client import LLMCallMetadata
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    GoldenCase,
    HumanDecision,
    ShadowEvaluationRecord,
    ShadowEvaluationSummary,
)


@dataclass(frozen=True)
class ShadowEvaluationGates:
    min_schema_success_rate: float = 0.98
    min_decision_accuracy: float = 0.95
    min_policy_accuracy: float = 0.98
    max_false_approve_rate: float = 0.0
    max_false_reject_rate: float = 0.0
    max_provider_count: int = 1
    max_response_model_count: int = 1

    @classmethod
    def from_json(cls, path: Path) -> "ShadowEvaluationGates":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload["thresholds"])

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in self.__dict__.items()
        }


def make_shadow_record(
    *,
    eval_run_id: str,
    case: GoldenCase,
    production_decision: AgentDecision,
    shadow_decision: AgentDecision,
    metadata: LLMCallMetadata,
    schema_valid: bool,
    latency_ms: int,
) -> ShadowEvaluationRecord:
    decision_correct = (
        shadow_decision.decision.value == case.human_decision.value
    )
    policy_correct = None
    if case.human_decision == HumanDecision.REJECT:
        policy_correct = shadow_decision.policy_label == case.expected_policy
    return ShadowEvaluationRecord(
        eval_run_id=eval_run_id,
        case_id=case.case_id,
        requested_model=metadata.requested_model,
        response_model=metadata.response_model,
        provider=metadata.provider,
        request_id=metadata.request_id,
        status_code=metadata.status_code,
        prompt_tokens=metadata.prompt_tokens,
        completion_tokens=metadata.completion_tokens,
        total_tokens=metadata.total_tokens,
        cost=metadata.cost,
        schema_valid=schema_valid,
        production_decision=production_decision.decision,
        shadow_decision=shadow_decision.decision,
        shadow_policy=shadow_decision.policy_label,
        decision_correct=decision_correct,
        policy_correct=policy_correct,
        confidence=shadow_decision.confidence,
        latency_ms=latency_ms,
        evidence_chunk_ids=[
            evidence.chunk_id
            for evidence in shadow_decision.matched_policy
        ],
        reason=shadow_decision.reason,
        split=case.split,
        tags=case.tags,
    )


def summarize_shadow(
    records: List[ShadowEvaluationRecord],
    eval_run_id: str,
    requested_model: str,
    gates: Optional[ShadowEvaluationGates] = None,
) -> ShadowEvaluationSummary:
    total = len(records)
    human_rejects = [
        record
        for record in records
        if record.shadow_policy is not None
        or record.policy_correct is not None
    ]
    human_approves = [
        record for record in records if record.policy_correct is None
    ]
    policy_records = [
        record
        for record in human_rejects
        if record.shadow_decision == AgentDecisionLabel.REJECT
    ]
    false_approves = sum(
        record.shadow_decision == AgentDecisionLabel.APPROVE
        for record in human_rejects
    )
    false_rejects = sum(
        record.shadow_decision == AgentDecisionLabel.REJECT
        for record in human_approves
    )
    provider_counts = Counter(
        record.provider or "unknown" for record in records
    )
    response_model_counts = Counter(
        record.response_model or "unknown" for record in records
    )
    summary = ShadowEvaluationSummary(
        eval_run_id=eval_run_id,
        requested_model=requested_model,
        total_cases=total,
        schema_valid_cases=sum(record.schema_valid for record in records),
        schema_success_rate=(
            sum(record.schema_valid for record in records) / total
            if total
            else 0.0
        ),
        decision_accuracy=(
            sum(record.decision_correct for record in records) / total
            if total
            else 0.0
        ),
        policy_accuracy=(
            sum(record.policy_correct is True for record in policy_records)
            / len(policy_records)
            if policy_records
            else 0.0
        ),
        false_approve_rate=(
            false_approves / len(human_rejects)
            if human_rejects
            else 0.0
        ),
        false_reject_rate=(
            false_rejects / len(human_approves)
            if human_approves
            else 0.0
        ),
        total_tokens=sum(record.total_tokens for record in records),
        total_cost=sum(record.cost or 0.0 for record in records),
        provider_counts=dict(sorted(provider_counts.items())),
        response_model_counts=dict(sorted(response_model_counts.items())),
    )
    gate_config = gates or ShadowEvaluationGates()
    comparisons = [
        (
            "schema_success_rate",
            summary.schema_success_rate,
            ">=",
            gate_config.min_schema_success_rate,
        ),
        (
            "decision_accuracy",
            summary.decision_accuracy,
            ">=",
            gate_config.min_decision_accuracy,
        ),
        (
            "policy_accuracy",
            summary.policy_accuracy,
            ">=",
            gate_config.min_policy_accuracy,
        ),
        (
            "false_approve_rate",
            summary.false_approve_rate,
            "<=",
            gate_config.max_false_approve_rate,
        ),
        (
            "false_reject_rate",
            summary.false_reject_rate,
            "<=",
            gate_config.max_false_reject_rate,
        ),
        (
            "provider_count",
            float(len(summary.provider_counts)),
            "<=",
            float(gate_config.max_provider_count),
        ),
        (
            "response_model_count",
            float(len(summary.response_model_counts)),
            "<=",
            float(gate_config.max_response_model_count),
        ),
    ]
    failures = []
    for name, actual, operator, threshold in comparisons:
        failed = actual < threshold if operator == ">=" else actual > threshold
        if failed:
            failures.append(
                f"{name}={actual:.4f} must be {operator} {threshold:.4f}"
            )
    if "unknown" in summary.provider_counts:
        failures.append("provider metadata must be present for every shadow call")
    if "unknown" in summary.response_model_counts:
        failures.append(
            "response model metadata must be present for every shadow call"
        )
    return summary.model_copy(
        update={
            "gates": gate_config.as_dict(),
            "failed_gates": failures,
        }
    )


def write_shadow_artifacts(
    records: Iterable[ShadowEvaluationRecord],
    output_directory: Path,
    eval_run_id: str,
    requested_model: str,
    gates: Optional[ShadowEvaluationGates] = None,
) -> Dict[str, Path]:
    record_list = list(records)
    output_directory.mkdir(parents=True, exist_ok=True)
    records_path = output_directory / "shadow_records.jsonl"
    report_path = output_directory / "shadow_report.json"
    summary = summarize_shadow(
        record_list,
        eval_run_id,
        requested_model,
        gates,
    )
    records_path.write_text(
        "\n".join(record.model_dump_json() for record in record_list) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return {"shadow_records": records_path, "shadow_report": report_path}

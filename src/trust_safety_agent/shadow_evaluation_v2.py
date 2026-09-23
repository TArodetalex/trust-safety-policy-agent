"""Truth-preserving metrics for new multimodal shadow runs.

The v1 module remains unchanged because it is part of the frozen Day 7
baseline. New runs use this version, which records human truth directly instead
of inferring it from model output.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from trust_safety_agent.llm_client import LLMCallMetadata
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    GoldenCase,
    HumanDecision,
    PolicyLabel,
    ShadowEvaluationRecord,
    ShadowEvaluationSummary,
)
from trust_safety_agent.shadow_evaluation import ShadowEvaluationGates


class ShadowEvaluationRecordV2(ShadowEvaluationRecord):
    record_version: str = "v2.0.0"
    human_decision: HumanDecision
    expected_policy: Optional[PolicyLabel] = None


def make_shadow_record_v2(
    *,
    eval_run_id: str,
    case: GoldenCase,
    production_decision: AgentDecision,
    shadow_decision: AgentDecision,
    metadata: LLMCallMetadata,
    schema_valid: bool,
    latency_ms: int,
) -> ShadowEvaluationRecordV2:
    decision_correct = shadow_decision.decision.value == case.human_decision.value
    policy_correct = None
    if case.human_decision == HumanDecision.REJECT:
        policy_correct = shadow_decision.policy_label == case.expected_policy
    return ShadowEvaluationRecordV2(
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
            evidence.chunk_id for evidence in shadow_decision.matched_policy
        ],
        reason=shadow_decision.reason,
        split=case.split,
        tags=case.tags,
        human_decision=case.human_decision,
        expected_policy=case.expected_policy,
    )


def summarize_shadow_v2(
    records: List[ShadowEvaluationRecordV2],
    eval_run_id: str,
    requested_model: str,
    gates: Optional[ShadowEvaluationGates] = None,
) -> ShadowEvaluationSummary:
    total = len(records)
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
    provider_counts = Counter(record.provider or "unknown" for record in records)
    response_model_counts = Counter(
        record.response_model or "unknown" for record in records
    )
    summary = ShadowEvaluationSummary(
        eval_run_id=eval_run_id,
        requested_model=requested_model,
        total_cases=total,
        schema_valid_cases=sum(record.schema_valid for record in records),
        schema_success_rate=(
            sum(record.schema_valid for record in records) / total if total else 0.0
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
            false_approves / len(human_rejects) if human_rejects else 0.0
        ),
        false_reject_rate=(
            false_rejects / len(human_approves) if human_approves else 0.0
        ),
        total_tokens=sum(record.total_tokens for record in records),
        total_cost=sum(record.cost or 0.0 for record in records),
        provider_counts=dict(sorted(provider_counts.items())),
        response_model_counts=dict(sorted(response_model_counts.items())),
    )
    gate_config = gates or ShadowEvaluationGates()
    checks = (
        ("schema_success_rate", summary.schema_success_rate, ">=", gate_config.min_schema_success_rate),
        ("decision_accuracy", summary.decision_accuracy, ">=", gate_config.min_decision_accuracy),
        ("policy_accuracy", summary.policy_accuracy, ">=", gate_config.min_policy_accuracy),
        ("false_approve_rate", summary.false_approve_rate, "<=", gate_config.max_false_approve_rate),
        ("false_reject_rate", summary.false_reject_rate, "<=", gate_config.max_false_reject_rate),
        ("provider_count", float(len(summary.provider_counts)), "<=", float(gate_config.max_provider_count)),
        ("response_model_count", float(len(summary.response_model_counts)), "<=", float(gate_config.max_response_model_count)),
    )
    failures = []
    for name, actual, operator, threshold in checks:
        failed = actual < threshold if operator == ">=" else actual > threshold
        if failed:
            failures.append(
                f"{name}={actual:.4f} must be {operator} {threshold:.4f}"
            )
    if "unknown" in summary.provider_counts:
        failures.append("provider metadata must be present for every shadow call")
    if "unknown" in summary.response_model_counts:
        failures.append("response model metadata must be present for every shadow call")
    return summary.model_copy(
        update={"gates": gate_config.as_dict(), "failed_gates": failures}
    )

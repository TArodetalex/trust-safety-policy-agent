from trust_safety_agent.failure_taxonomy import (
    aggregate_failures,
    classify_evaluation_failure,
)
from trust_safety_agent.schema import (
    AgentDecisionLabel,
    ErrorType,
    EvaluationRecord,
    FailureType,
    HumanDecision,
)


def _record(error_type: ErrorType, evidence: bool = False) -> EvaluationRecord:
    return EvaluationRecord(
        eval_run_id="EV-FAILURE-TEST",
        case_id="C001",
        prompt_version="test",
        retrieval_version="test",
        human_decision=HumanDecision.REJECT,
        agent_decision=AgentDecisionLabel.APPROVE,
        is_decision_correct=False,
        error_type=error_type,
        evidence_chunk_ids=["PCH-000000000001"] if evidence else [],
        latency_ms=1,
    )


def test_failure_taxonomy_maps_existing_evaluation_errors() -> None:
    assert (
        classify_evaluation_failure(_record(ErrorType.INVALID_OUTPUT))
        == FailureType.SCHEMA_ERROR
    )
    assert (
        classify_evaluation_failure(_record(ErrorType.FALSE_APPROVE))
        == FailureType.EVIDENCE_MISS
    )
    assert (
        classify_evaluation_failure(_record(ErrorType.FALSE_APPROVE, evidence=True))
        == FailureType.POLICY_MISREAD
    )


def test_failure_aggregation_counts_failure_and_error_dimensions() -> None:
    records = [
        _record(ErrorType.FALSE_APPROVE),
        _record(ErrorType.INVALID_OUTPUT),
    ]
    aggregation = aggregate_failures(records)

    assert aggregation.total_error_cases == 2
    assert aggregation.failure_counts == {
        "evidence_miss": 1,
        "schema_error": 1,
    }
    assert aggregation.by_error_type["false_approve"]["evidence_miss"] == 1


def test_unexpected_image_review_is_an_evidence_miss() -> None:
    record = _record(ErrorType.UNEXPECTED_REVIEW)
    record = record.model_copy(
        update={"agent_reason": "The rules engine cannot inspect image evidence safely."}
    )

    assert classify_evaluation_failure(record) == FailureType.EVIDENCE_MISS

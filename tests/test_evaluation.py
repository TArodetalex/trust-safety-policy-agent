from __future__ import annotations

import json
from pathlib import Path

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.dataset import load_cases, validate_dataset
from trust_safety_agent.evaluation import (
    EvaluationGates,
    evaluate_cases,
    write_evaluation_artifacts,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ErrorType,
    ExpectedRoute,
    GoldenCase,
    MatchedPolicyEvidence,
    PolicyLabel,
)
from trust_safety_agent.vector_store import PolicyVectorStore


ROOT = Path(__file__).parents[1]
DATASET_V2 = ROOT / "data/golden_set/golden_set_v2.csv"
POLICY_PATH = ROOT / "data/policies/mock_policy_v1.md"


def make_case(
    case_id: str,
    human_decision: str,
    expected_route: str = "auto_decide",
) -> GoldenCase:
    return GoldenCase(
        dataset_version="v2.0.0",
        case_id=case_id,
        content_type="product",
        input_text=f"Evaluation case {case_id}",
        human_decision=human_decision,
        expected_policy=(
            PolicyLabel.COUNTERFEIT if human_decision == "reject" else None
        ),
        expected_exemption="none",
        risk_level="high",
        is_boundary_case=True,
        human_reason="Human ground truth for the evaluation case.",
        expected_route=expected_route,
        tags=["unit_test"],
    )


def make_decision(
    case_id: str,
    decision: AgentDecisionLabel,
) -> AgentDecision:
    if decision == AgentDecisionLabel.REJECT:
        return AgentDecision(
            case_id=case_id,
            decision=decision,
            policy_label=PolicyLabel.COUNTERFEIT,
            matched_policy=[
                MatchedPolicyEvidence(
                    policy_id="POL-CF-001",
                    chunk_id="PCH-123456789abc",
                    quote="Counterfeit goods are prohibited.",
                )
            ],
            reason="The policy directly supports rejection.",
            recommended_action="Reject.",
            confidence=0.95,
        )
    return AgentDecision(
        case_id=case_id,
        decision=decision,
        reason="No automatic rejection is supported.",
        recommended_action=(
            "Approve." if decision == AgentDecisionLabel.APPROVE else "Review."
        ),
        confidence=0.8,
    )


def test_v2_dataset_is_valid_and_has_review_cases() -> None:
    cases = load_cases(DATASET_V2)

    assert validate_dataset(cases) == []
    assert len(cases) == 70
    assert sum(
        case.expected_route == ExpectedRoute.HUMAN_REVIEW for case in cases
    ) == 10
    assert all(case.tags for case in cases)


def test_selective_metrics_separate_accuracy_and_routing() -> None:
    cases = [
        make_case("C901", "approve"),
        make_case("C902", "reject"),
        make_case("C903", "approve", "human_review"),
        make_case("C904", "reject"),
    ]
    decisions = {
        "C901": make_decision("C901", AgentDecisionLabel.APPROVE),
        "C902": make_decision("C902", AgentDecisionLabel.REJECT),
        "C903": make_decision("C903", AgentDecisionLabel.NEED_REVIEW),
        "C904": make_decision("C904", AgentDecisionLabel.APPROVE),
    }

    report, records = evaluate_cases(
        cases,
        lambda case: decisions[case.case_id],
        engine="test",
        prompt_version="test-v1",
        retrieval_version="test-index",
        eval_run_id="EV-TEST",
    )

    assert report.metrics.accuracy == 0.5
    assert report.metrics.auto_accuracy == 2 / 3
    assert report.metrics.coverage == 0.75
    assert report.metrics.route_accuracy == 1.0
    assert report.metrics.false_approve_rate == 0.5
    assert report.metrics.policy_accuracy == 1.0
    assert report.metrics.confusion_matrix["approve"]["need_review"] == 1
    assert report.metrics.error_counts == {"false_approve": 1}
    assert report.slices["split:regression"].total_cases == 4
    assert records[2].error_type == ErrorType.NONE


def test_unexpected_review_is_bucketed() -> None:
    case = make_case("C905", "approve")

    report, records = evaluate_cases(
        [case],
        lambda item: make_decision(
            item.case_id,
            AgentDecisionLabel.NEED_REVIEW,
        ),
        engine="test",
        prompt_version="test-v1",
        retrieval_version="test-index",
        eval_run_id="EV-REVIEW",
    )

    assert report.metrics.route_accuracy == 0
    assert records[0].error_type == ErrorType.UNEXPECTED_REVIEW


def test_provider_exception_becomes_invalid_output() -> None:
    case = make_case("C906", "reject")

    def fail(_: GoldenCase) -> AgentDecision:
        raise RuntimeError("provider failed")

    _, records = evaluate_cases(
        [case],
        fail,
        engine="test",
        prompt_version="test-v1",
        retrieval_version="test-index",
        eval_run_id="EV-ERROR",
    )

    assert records[0].error_type == ErrorType.INVALID_OUTPUT
    assert "provider failed" in records[0].agent_reason


def test_artifacts_are_written(tmp_path: Path) -> None:
    case = make_case("C907", "approve")
    report, records = evaluate_cases(
        [case],
        lambda item: make_decision(item.case_id, AgentDecisionLabel.APPROVE),
        engine="test",
        prompt_version="test-v1",
        retrieval_version="test-index",
        eval_run_id="EV-WRITE",
    )

    paths = write_evaluation_artifacts(report, records, tmp_path)

    payload = json.loads(paths["report"].read_text())
    assert payload["eval_run_id"] == "EV-WRITE"
    assert paths["records"].read_text().count("\n") == 1
    assert paths["errors"].read_text().startswith("case_id,error_type")


def test_offline_v2_baseline_passes_default_gates(tmp_path: Path) -> None:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(POLICY_PATH))
    agent = PolicyAdjudicator(store)
    cases = load_cases(DATASET_V2)

    report, _ = evaluate_cases(
        cases,
        lambda case: agent.adjudicate(
            case.case_id,
            case.content_type,
            case.input_text,
        ),
        engine="rules",
        prompt_version="rules-v1",
        retrieval_version=store.collection_name,
        gates=EvaluationGates(),
        eval_run_id="EV-BASELINE",
    )

    assert report.failed_gates == []
    assert report.metrics.auto_accuracy == 1
    assert report.metrics.route_accuracy == 1
    assert report.metrics.false_approve_rate == 0
    assert report.metrics.false_reject_rate == 0

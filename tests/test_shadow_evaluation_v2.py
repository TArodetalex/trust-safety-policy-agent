from trust_safety_agent.llm_client import LLMCallMetadata
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    GoldenCase,
    MatchedPolicyEvidence,
    PolicyLabel,
)
from trust_safety_agent.shadow_evaluation_v2 import (
    make_shadow_record_v2,
    summarize_shadow_v2,
)


def _case() -> GoldenCase:
    return GoldenCase(
        dataset_version="v4.0.0",
        case_id="C171",
        content_type="product",
        input_text="Visual shadow case.",
        image_path="data/golden_set/assets/v4/IMG-C171.png",
        human_decision="reject",
        expected_policy="counterfeit",
        risk_level="high",
        is_boundary_case=True,
        human_reason="Synthetic shadow evaluation truth.",
        split="blind",
        tags=["shadow"],
    )


def _decision(label: AgentDecisionLabel) -> AgentDecision:
    if label == AgentDecisionLabel.REJECT:
        return AgentDecision(
            case_id="C171",
            decision=label,
            policy_label=PolicyLabel.COUNTERFEIT,
            matched_policy=[
                MatchedPolicyEvidence(
                    policy_id="POL-CF-001",
                    chunk_id="PCH-123456789abc",
                    quote="Counterfeit promotion is prohibited.",
                )
            ],
            reason="Explicit counterfeit evidence is visible.",
            recommended_action="Reject.",
            confidence=0.96,
        )
    return AgentDecision(
        case_id="C171",
        decision=label,
        reason="No violation is visible.",
        recommended_action="Approve.",
        confidence=0.92,
    )


def test_false_approve_uses_human_truth() -> None:
    record = make_shadow_record_v2(
        eval_run_id="EV-SHADOW-V2-TEST",
        case=_case(),
        production_decision=_decision(AgentDecisionLabel.REJECT),
        shadow_decision=_decision(AgentDecisionLabel.APPROVE),
        metadata=LLMCallMetadata(
            requested_model="fixed-model",
            response_model="fixed-model",
            provider="test-provider",
        ),
        schema_valid=True,
        latency_ms=10,
    )

    summary = summarize_shadow_v2(
        [record],
        "EV-SHADOW-V2-TEST",
        "fixed-model",
    )

    assert record.human_decision.value == "reject"
    assert summary.false_approve_rate == 1.0
    assert summary.false_reject_rate == 0.0
    assert any("false_approve_rate" in failure for failure in summary.failed_gates)

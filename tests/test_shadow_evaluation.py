from __future__ import annotations

import json

from trust_safety_agent.llm_client import LLMCallMetadata
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    GoldenCase,
    MatchedPolicyEvidence,
    PolicyLabel,
)
from trust_safety_agent.shadow_evaluation import (
    make_shadow_record,
    summarize_shadow,
    write_shadow_artifacts,
)


def golden_case(case_id: str, decision: str) -> GoldenCase:
    return GoldenCase(
        dataset_version="v4.0.0",
        case_id=case_id,
        content_type="product",
        input_text="Visual shadow case.",
        image_path=f"data/golden_set/assets/v4/IMG-{case_id}.png",
        human_decision=decision,
        expected_policy=(
            PolicyLabel.COUNTERFEIT if decision == "reject" else None
        ),
        expected_exemption="none",
        risk_level="high" if decision == "reject" else "low",
        is_boundary_case=True,
        human_reason="Synthetic shadow evaluation truth.",
        split="blind",
        expected_route="auto_decide",
        tags=["day7", "multimodal", "shadow"],
    )


def agent_decision(case_id: str, decision: AgentDecisionLabel) -> AgentDecision:
    if decision == AgentDecisionLabel.REJECT:
        return AgentDecision(
            case_id=case_id,
            decision=decision,
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
        case_id=case_id,
        decision=decision,
        reason="No violation is visible.",
        recommended_action="Approve.",
        confidence=0.92,
    )


def test_shadow_summary_tracks_schema_accuracy_and_cost(tmp_path) -> None:
    metadata = LLMCallMetadata(
        requested_model="openai/gpt-4o",
        response_model="openai/gpt-4o-2024-11-20",
        provider="OpenAI",
        status_code=200,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cost=0.01,
    )
    approve_case = golden_case("C201", "approve")
    reject_case = golden_case("C171", "reject")
    records = [
        make_shadow_record(
            eval_run_id="EV-DAY7-TEST",
            case=approve_case,
            production_decision=agent_decision(
                "C201", AgentDecisionLabel.NEED_REVIEW
            ),
            shadow_decision=agent_decision(
                "C201", AgentDecisionLabel.APPROVE
            ),
            metadata=metadata,
            schema_valid=True,
            latency_ms=100,
        ),
        make_shadow_record(
            eval_run_id="EV-DAY7-TEST",
            case=reject_case,
            production_decision=agent_decision(
                "C171", AgentDecisionLabel.REJECT
            ),
            shadow_decision=agent_decision(
                "C171", AgentDecisionLabel.REJECT
            ),
            metadata=metadata,
            schema_valid=True,
            latency_ms=120,
        ),
    ]

    summary = summarize_shadow(records, "EV-DAY7-TEST", "openai/gpt-4o")

    assert summary.total_cases == 2
    assert summary.schema_success_rate == 1
    assert summary.decision_accuracy == 1
    assert summary.policy_accuracy == 1
    assert summary.total_tokens == 240
    assert summary.total_cost == 0.02
    assert summary.provider_counts == {"OpenAI": 2}

    paths = write_shadow_artifacts(
        records,
        tmp_path,
        "EV-DAY7-TEST",
        "openai/gpt-4o",
    )
    report = json.loads(paths["shadow_report"].read_text())
    assert report["total_cases"] == 2
    assert paths["shadow_records"].read_text().count("\n") == 2

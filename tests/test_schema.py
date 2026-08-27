from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.validate_golden_set import load_cases, validate_dataset
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ExemptionType,
    GoldenCase,
    HumanDecision,
    PolicyLabel,
)


DATASET = Path(__file__).parents[1] / "data/golden_set/golden_set_v1.csv"


def base_case() -> dict[str, object]:
    return {
        "dataset_version": "v1.0.0",
        "case_id": "C999",
        "content_type": "product",
        "input_text": "Generic product",
        "human_decision": "approve",
        "expected_exemption": "none",
        "risk_level": "low",
        "is_boundary_case": False,
        "human_reason": "No violation signal is present.",
    }


def test_golden_set_is_valid_and_complete() -> None:
    cases = load_cases(DATASET)

    assert validate_dataset(cases) == []
    assert len(cases) == 50
    assert {case.expected_policy for case in cases if case.expected_policy} == set(
        PolicyLabel
    )
    assert {
        case.expected_exemption
        for case in cases
        if case.expected_exemption != ExemptionType.NONE
    } == set(ExemptionType) - {ExemptionType.NONE}


def test_reject_case_requires_policy() -> None:
    payload = base_case() | {"human_decision": HumanDecision.REJECT}

    with pytest.raises(ValidationError, match="require expected_policy"):
        GoldenCase.model_validate(payload)


def test_approve_case_cannot_declare_policy() -> None:
    payload = base_case() | {"expected_policy": PolicyLabel.COUNTERFEIT}

    with pytest.raises(ValidationError, match="cannot declare expected_policy"):
        GoldenCase.model_validate(payload)


def test_exemption_case_must_be_boundary() -> None:
    payload = base_case() | {"expected_exemption": ExemptionType.COMPATIBILITY}

    with pytest.raises(ValidationError, match="must be marked as boundary"):
        GoldenCase.model_validate(payload)


def test_reject_decision_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="matched policy evidence"):
        AgentDecision(
            case_id="C001",
            decision=AgentDecisionLabel.REJECT,
            policy_label=PolicyLabel.COUNTERFEIT,
            reason="The listing explicitly says replica.",
            recommended_action="Reject the listing.",
            confidence=0.98,
        )

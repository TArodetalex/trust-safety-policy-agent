from pathlib import Path

import pytest

from trust_safety_agent.product_review import ProductReviewDecision, ReviewConfidence
from trust_safety_agent.reviewer_workspace import (
    ReviewerFeedback,
    ReviewerLabel,
    ReviewerWorkspaceStore,
    ReviewWorkflow,
)


def test_reviewer_decision_does_not_overwrite_agent_decision(tmp_path: Path) -> None:
    store = ReviewerWorkspaceStore(tmp_path / "reviews.jsonl")
    queued = store.enqueue(
        case_id="PR-100",
        workflow=ReviewWorkflow.PRODUCT_IPR,
        case_input={"title": "Gucci bag"},
        agent_decision=ProductReviewDecision.MANUAL_REVIEW,
        agent_confidence=ReviewConfidence.LOW,
        agent_reason="Evidence is incomplete.",
    )
    completed = store.complete(
        queued.review_id,
        ReviewerFeedback(
            reviewer_label=ReviewerLabel.REJECT,
            reviewer_note="Confirmed counterfeit wording and brand evidence.",
            add_to_golden_set=True,
            expected_policy="counterfeit",
            expected_evidence=["replica"],
            case_category="product_ipr",
            failure_type="evidence_miss",
        ),
    )

    assert completed.agent_decision == ProductReviewDecision.MANUAL_REVIEW
    assert completed.reviewer_label == ReviewerLabel.REJECT
    assert completed.final_decision == ProductReviewDecision.REJECT
    assert store.list_records()[0] == completed


def test_uncertain_review_cannot_enter_staging() -> None:
    with pytest.raises(ValueError, match="uncertain"):
        ReviewerFeedback(
            reviewer_label=ReviewerLabel.UNCERTAIN,
            reviewer_note="Still missing the source image.",
            add_to_golden_set=True,
        )


def test_duplicate_pending_case_is_rejected(tmp_path: Path) -> None:
    store = ReviewerWorkspaceStore(tmp_path / "reviews.jsonl")
    arguments = dict(
        case_id="SHOP-100",
        workflow=ReviewWorkflow.SHOP_IDENTITY,
        case_input={"shop_name": "Gucci"},
        agent_decision=ProductReviewDecision.MANUAL_REVIEW,
        agent_confidence=ReviewConfidence.LOW,
        agent_reason="Authorization is unknown.",
    )
    store.enqueue(**arguments)

    with pytest.raises(ValueError, match="pending"):
        store.enqueue(**arguments)


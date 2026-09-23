import json
from pathlib import Path

import pytest

from trust_safety_agent.product_review import ProductReviewDecision, ReviewConfidence
from trust_safety_agent.reviewer_workspace import (
    ReviewerFeedback,
    ReviewerLabel,
    ReviewerWorkspaceStore,
    ReviewWorkflow,
)
from trust_safety_agent.staging_dataset import StagingDatasetStore, StagingStatus


def _completed_review(tmp_path: Path):
    reviews = ReviewerWorkspaceStore(tmp_path / "reviews.jsonl")
    queued = reviews.enqueue(
        case_id="PR-200",
        workflow=ReviewWorkflow.PRODUCT_IPR,
        case_input={"title": "Replica Gucci bag"},
        agent_decision=ProductReviewDecision.REJECT,
        agent_confidence=ReviewConfidence.HIGH,
        agent_reason="Explicit replica signal.",
    )
    return reviews.complete(
        queued.review_id,
        ReviewerFeedback(
            reviewer_label=ReviewerLabel.REJECT,
            reviewer_note="Human verified the evidence.",
            add_to_golden_set=True,
            expected_policy="counterfeit",
            expected_evidence=["Replica Gucci bag"],
            case_category="product_ipr",
        ),
    )


def test_review_enters_staging_and_requires_explicit_promotion(tmp_path: Path) -> None:
    staging = StagingDatasetStore(tmp_path / "staging" / "candidates.jsonl")
    candidate = staging.add_review(_completed_review(tmp_path))

    assert candidate.status == StagingStatus.CANDIDATE
    assert not (tmp_path / "v5" / "golden_set_v5.jsonl").exists()

    manifest = staging.promote(tmp_path / "v5")

    assert manifest.dataset_version == "v5.0.0"
    assert manifest.record_count == 1
    assert len(manifest.sha256) == 64
    assert staging.list_records()[0].status == StagingStatus.PROMOTED
    saved_manifest = json.loads(
        (tmp_path / "v5" / "golden_set_v5_manifest.json").read_text("utf-8")
    )
    assert saved_manifest["sha256"] == manifest.sha256


def test_same_review_cannot_be_added_twice(tmp_path: Path) -> None:
    review = _completed_review(tmp_path)
    staging = StagingDatasetStore(tmp_path / "staging.jsonl")
    staging.add_review(review)

    with pytest.raises(ValueError, match="already exists"):
        staging.add_review(review)

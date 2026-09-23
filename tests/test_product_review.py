from pathlib import Path

import pytest
from pydantic import ValidationError

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_POLICY_PATH
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.product_review import (
    ProductReviewAssistant,
    ProductReviewDecision,
    ProductReviewInput,
    ReviewConfidence,
)
from trust_safety_agent.schema import ExemptionType
from trust_safety_agent.vector_store import PolicyVectorStore


@pytest.fixture
def assistant(tmp_path: Path) -> ProductReviewAssistant:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    library = ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)
    return ProductReviewAssistant(library, store)


def test_counterfeit_brand_is_rejected_with_policy_evidence(
    assistant: ProductReviewAssistant,
) -> None:
    result = assistant.review(
        ProductReviewInput(
            case_id="PR-001",
            title="Gucci handbag",
            description="1:1 mirror copy with branded dust bag",
        )
    )

    assert result.suggested_decision == ProductReviewDecision.REJECT
    assert result.confidence == ReviewConfidence.HIGH
    assert result.controlled_brand_matches == ["Gucci"]
    assert any(ref.policy_id == "POL-CF-001" for ref in result.policy_references)


def test_compatibility_context_is_approved_as_exemption(
    assistant: ProductReviewAssistant,
) -> None:
    result = assistant.review(
        ProductReviewInput(
            case_id="PR-002",
            title="USB-C charging cable",
            description="Compatible with Apple iPhone 15 charging ports",
        )
    )

    assert result.suggested_decision == ProductReviewDecision.APPROVE
    assert ExemptionType.COMPATIBILITY in result.possible_exemptions
    assert "Apple" in result.controlled_brand_matches


def test_common_short_brand_requires_context_review(
    assistant: ProductReviewAssistant,
) -> None:
    result = assistant.review(
        ProductReviewInput(
            case_id="PR-003",
            title="GE replacement part",
        )
    )

    assert result.suggested_decision == ProductReviewDecision.MANUAL_REVIEW
    assert result.confidence == ReviewConfidence.LOW
    assert result.candidate_brands[0].ambiguity is True


def test_image_without_visual_evidence_routes_to_review(
    assistant: ProductReviewAssistant,
) -> None:
    result = assistant.review(
        ProductReviewInput(
            case_id="PR-004",
            title="Leather handbag",
            image_url="https://example.com/product.jpg",
        )
    )

    assert result.suggested_decision == ProductReviewDecision.MANUAL_REVIEW
    assert "Logo" in result.reviewer_checkpoints[0]


def test_product_input_rejects_conflicting_image_sources() -> None:
    with pytest.raises(ValidationError):
        ProductReviewInput(
            case_id="PR-005",
            title="Test product",
            image_url="https://example.com/product.jpg",
            image_path="uploads/product.jpg",
        )


def test_product_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ProductReviewInput(
            case_id="PR-006",
            title="Test product",
            unexpected="value",
        )

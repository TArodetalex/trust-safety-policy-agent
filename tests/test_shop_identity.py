from pathlib import Path

import pytest

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_POLICY_PATH
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.product_review import ProductReviewDecision
from trust_safety_agent.shop_identity import (
    AuthorizationStatus,
    AvatarSignalType,
    ShopIdentityInput,
    ShopIdentityReviewer,
    ShopNameSignalType,
)
from trust_safety_agent.vector_store import PolicyVectorStore


@pytest.fixture
def reviewer(tmp_path: Path) -> ShopIdentityReviewer:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    library = ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)
    return ShopIdentityReviewer(library, store)


def test_authorized_shop_is_approved_before_identity_risk(
    reviewer: ShopIdentityReviewer,
) -> None:
    result = reviewer.review(
        ShopIdentityInput(
            case_id="SHOP-001",
            shop_name="Gucci Official Flagship Store",
            avatar_url="https://example.com/gucci.png",
            controlled_brand="Gucci",
            authorization_status=AuthorizationStatus.AUTHORIZED,
            avatar_visual_marks=["Gucci logo"],
        )
    )

    assert result.suggested_decision == ProductReviewDecision.APPROVE
    assert result.policy_references == []
    assert result.possible_exemptions == ["documented_authorization"]


def test_unauthorized_impersonation_is_rejected_with_policy(
    reviewer: ShopIdentityReviewer,
) -> None:
    result = reviewer.review(
        ShopIdentityInput(
            case_id="SHOP-002",
            shop_name="Gucci Official Store",
            controlled_brand="Gucci",
            authorization_status=AuthorizationStatus.UNAUTHORIZED,
        )
    )

    assert result.shop_name_signal.signal_type == ShopNameSignalType.IMPERSONATION
    assert result.suggested_decision == ProductReviewDecision.REJECT
    assert any(item.policy_id == "POL-SI-001" for item in result.policy_references)


def test_unknown_authorization_and_unavailable_avatar_route_to_review(
    reviewer: ShopIdentityReviewer,
) -> None:
    result = reviewer.review(
        ShopIdentityInput(
            case_id="SHOP-003",
            shop_name="Gucci",
            avatar_url="https://example.com/avatar.png",
            controlled_brand="Gucci",
        )
    )

    assert result.avatar_signal.signal_type == AvatarSignalType.UNAVAILABLE_IMAGE
    assert result.suggested_decision == ProductReviewDecision.MANUAL_REVIEW
    assert any("Logo" in item for item in result.reviewer_checkpoints)


def test_common_word_context_is_not_identity_evidence(
    reviewer: ShopIdentityReviewer,
) -> None:
    signal = reviewer.judge_shop_name(
        "Apple Juice Market",
        reviewer.library.get("Apple"),
    )

    assert signal.signal_type == ShopNameSignalType.MEANINGFUL_COMMON_WORD


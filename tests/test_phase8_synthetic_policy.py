import csv
from pathlib import Path

import pytest

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import (
    DEFAULT_BRAND_LIBRARY_PATH,
    DEFAULT_DEMO_ASSET_MANIFEST_PATH,
    DEFAULT_GOLDEN_SET_V4_PATH,
    DEFAULT_POLICY_PATH,
    DEFAULT_SYNTHETIC_DEMO_CASES_PATH,
    PROJECT_ROOT,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.product_review import ProductReviewAssistant, ProductReviewInput
from trust_safety_agent.shop_identity import ShopIdentityInput, ShopIdentityReviewer
from trust_safety_agent.synthetic_dataset import (
    DemoWorkflow,
    load_synthetic_demo_cases,
    verify_asset_manifest,
)
from trust_safety_agent.vector_store import PolicyVectorStore


@pytest.fixture
def phase8_dependencies(tmp_path: Path):
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    library = ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)
    return store, library


def test_synthetic_policy_v2_has_required_public_taxonomy() -> None:
    chunks = load_policy_file(DEFAULT_POLICY_PATH)
    policy_ids = {chunk.policy_id for chunk in chunks}

    assert DEFAULT_POLICY_PATH.name == "synthetic_policy_v2.md"
    assert all(chunk.policy_version == "v2.0.0" for chunk in chunks)
    assert {
        "POL-CF-001",
        "POL-KO-001",
        "POL-MBA-001",
        "POL-TMI-001",
        "POL-SI-001",
        "POL-EX-001",
        "POL-ER-001",
    }.issubset(policy_ids)


def test_v5_candidate_assets_are_synthetic_and_integrity_checked() -> None:
    cases = load_synthetic_demo_cases(DEFAULT_SYNTHETIC_DEMO_CASES_PATH)

    assert len(cases) == 14
    assert len({item.image_path for item in cases}) == 12
    assert {item.source for item in cases} == {"synthetic_phase8"}
    assert {item.split for item in cases} == {"candidate"}
    assert verify_asset_manifest(PROJECT_ROOT, DEFAULT_DEMO_ASSET_MANIFEST_PATH) == []


def test_v4_frozen_baseline_remains_separate_from_v5_candidates() -> None:
    with DEFAULT_GOLDEN_SET_V4_PATH.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == 210
    assert all(not row["case_id"].startswith("CASE-V5-") for row in rows)
    assert DEFAULT_SYNTHETIC_DEMO_CASES_PATH.parent.name == "v5_candidates"


def test_all_v5_candidate_decisions_match_declared_expectations(
    phase8_dependencies,
) -> None:
    store, library = phase8_dependencies
    product_reviewer = ProductReviewAssistant(library, store)
    shop_reviewer = ShopIdentityReviewer(library, store)

    for case in load_synthetic_demo_cases(DEFAULT_SYNTHETIC_DEMO_CASES_PATH):
        if case.workflow == DemoWorkflow.PRODUCT_IPR:
            payload = {
                key: value
                for key, value in case.model_dump().items()
                if key in ProductReviewInput.model_fields
            }
            result = product_reviewer.review(ProductReviewInput(**payload))
            assert result.risk_subtype == case.expected_risk_subtype
        else:
            result = shop_reviewer.review(
                ShopIdentityInput(
                    case_id=case.case_id,
                    shop_name=case.shop_name or "",
                    controlled_brand=case.controlled_brand or "",
                    authorization_status=case.brand_authorization_status.value,
                    avatar_visual_marks=case.visual_marks,
                )
            )

        assert result.suggested_decision == case.expected_decision, case.case_id
        if case.expected_policy_id:
            assert any(
                reference.policy_id == case.expected_policy_id
                for reference in result.policy_references
            ), case.case_id

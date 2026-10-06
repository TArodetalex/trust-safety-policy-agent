from pathlib import Path

from trust_safety_agent.brand_knowledge import BrandKnowledgeStore, load_brand_knowledge
from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.case_store import LabelStatus, ProductCase, RunDecision, ShopCase
from trust_safety_agent.config import DEFAULT_BRAND_KNOWLEDGE_PATH, DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_POLICY_PATH
from trust_safety_agent.phase9_workflow import (
    LabelAssessment,
    ProductRuntimeEvidence,
    ProductWorkflow,
    ReviewRouter,
    ShopItemStatus,
    ShopRuntimeEvidence,
    ShopWorkflow,
    aggregate_product_labels,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.price_search import OfflineFixturePriceProvider
from trust_safety_agent.vector_store import PolicyVectorStore


def build_workflows(tmp_path: Path):
    policy = PolicyVectorStore(tmp_path / "db", collection_name="policy_phase9_test")
    policy.index(load_policy_file(DEFAULT_POLICY_PATH))
    brand = BrandKnowledgeStore(tmp_path / "db")
    brand.index(load_brand_knowledge(DEFAULT_BRAND_KNOWLEDGE_PATH))
    library = ControlledBrandLibrary.from_csv_and_knowledge(DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_BRAND_KNOWLEDGE_PATH)
    return (
        ProductWorkflow(policy, brand, library, OfflineFixturePriceProvider()),
        ShopWorkflow(policy, brand, library),
    )


def test_product_multilabel_counterfeit_mba_and_trace(tmp_path: Path) -> None:
    product_workflow, _ = build_workflows(tmp_path)
    case = ProductCase(product_id="P-CF", title="Nike Air Force 1 replica", description="fake product", product_images=["data/demo_assets/v5/CASE-V5-007.png"], price=20, currency="USD", brand="Nike", brand_authorized=False)
    evidence = ProductRuntimeEvidence(detected_brand="Nike", detected_product="Air Force 1", observed_product_brand="Nike", logo_match="exact", fake_risk_evidence=True, image_readable=True)
    result, run, trace = product_workflow.run(case, evidence)
    assert result.labels["Counterfeit"].status == LabelStatus.HIT
    assert result.labels["Knockoff"].status == LabelStatus.NOT_HIT
    assert result.labels["MBA"].status == LabelStatus.HIT
    assert result.suggested_decision == RunDecision.REJECT
    assert run.trace_id == trace.trace_id
    names = [node.node_name for node in trace.nodes]
    assert names[:6] == ["validate_input", "normalize_and_extract_evidence", "recall_entities", "retrieve_policy", "retrieve_brand_knowledge", "retrieve_reference_price"]
    assert names[-4:] == ["evaluate_rules", "aggregate_decision", "apply_guardrail", "publish_result"]
    assert trace.nodes[0].started_at is not None and trace.nodes[0].ended_at is not None


def test_knockoff_mba_tmi_combination_and_missing_authorization_notice(tmp_path: Path) -> None:
    product_workflow, _ = build_workflows(tmp_path)
    case = ProductCase(product_id="P-KO", title="Nike style shoe", description="Nike branded listing", product_images=["data/demo_assets/v5/CASE-V5-002.png"], brand="Nike", brand_authorized=None)
    evidence = ProductRuntimeEvidence(detected_brand="Nike", observed_product_brand="unbranded", logo_match="modified", distinctive_design=False)
    result, _, _ = product_workflow.run(case, evidence)
    assert result.labels["Knockoff"].status == LabelStatus.HIT
    assert result.labels["MBA"].status == LabelStatus.HIT
    assert result.labels["TMI"].status == LabelStatus.HIT
    assert "provisional_missing_authorization" in result.labels["MBA"].qualifiers
    assert result.notices == ["请补充品牌授权情况。"]


def test_uncontrolled_brand_can_hit_counterfeit_but_not_mba_or_tmi(tmp_path: Path) -> None:
    product_workflow, _ = build_workflows(tmp_path)
    product_workflow.price_provider = OfflineFixturePriceProvider({"localbrand shoe": [100.0, 110.0]})
    case = ProductCase(product_id="P-U", title="LocalBrand shoe replica", product_images=["x.png"], price=10, brand="LocalBrand", brand_authorized=False)
    result, _, _ = product_workflow.run(case, ProductRuntimeEvidence(detected_brand="LocalBrand", detected_product="shoe", observed_product_brand="LocalBrand", logo_match="exact", fake_risk_evidence=True))
    assert result.labels["Counterfeit"].status == LabelStatus.HIT
    assert result.labels["MBA"].status == LabelStatus.NOT_APPLICABLE
    assert result.labels["TMI"].status == LabelStatus.NOT_APPLICABLE


def test_missing_reference_price_cannot_confirm_counterfeit(tmp_path: Path) -> None:
    product_workflow, _ = build_workflows(tmp_path)
    case = ProductCase(product_id="P-NP", title="Nike unknown replica", product_images=["x.png"], price=10, brand="Nike", brand_authorized=False)
    result, _, trace = product_workflow.run(case, ProductRuntimeEvidence(detected_brand="Nike", detected_product="unknown", observed_product_brand="Nike", logo_match="exact", fake_risk_evidence=True))
    assert result.labels["Counterfeit"].status == LabelStatus.INSUFFICIENT_EVIDENCE
    price_node = next(item for item in trace.nodes if item.node_name == "retrieve_reference_price")
    assert price_node.output["provider"] == "demo_fixture"


def test_conflicting_counterfeit_and_knockoff_is_guarded() -> None:
    labels = {
        "Counterfeit": LabelAssessment(status=LabelStatus.HIT, reason="exact"),
        "Knockoff": LabelAssessment(status=LabelStatus.HIT, reason="modified"),
        "MBA": LabelAssessment(status=LabelStatus.NOT_HIT, reason="no"),
        "TMI": LabelAssessment(status=LabelStatus.NOT_HIT, reason="no"),
    }
    assert aggregate_product_labels(labels) == (RunDecision.MANUAL_REVIEW, True)


def test_shop_workflow_keeps_name_avatar_separate_and_router_splits(tmp_path: Path) -> None:
    product_workflow, shop_workflow = build_workflows(tmp_path)
    shop = ShopCase(shop_id="S-1", shop_name="Nike Official Store", shop_avatar="avatar.png", brand="Nike", brand_authorized=False)
    evidence = ShopRuntimeEvidence(detected_name_brand="Nike", detected_avatar_brand=None, avatar_readable=True, official_identity_signal=True)
    shop_result, _, trace = shop_workflow.run(shop, evidence)
    assert shop_result.shop_name_status == ShopItemStatus.VIOLATION
    assert shop_result.shop_avatar_status == ShopItemStatus.COMPLIANT
    assert len(trace.nodes) == 9

    product = ProductCase(product_id="P-1", title="plain item", product_images=["image.png"])
    session = ReviewRouter(shop_workflow, product_workflow).run(shop_case=shop, product_case=product, shop_evidence=evidence)
    assert session.shop_result.case_id == "S-1"
    assert session.product_result.case_id == "P-1"
    assert session.shop_result.run_id != session.product_result.run_id
    assert session.route_run_id is not None
    assert session.route_trace_id is not None

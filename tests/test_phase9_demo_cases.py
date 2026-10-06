from pathlib import Path

from trust_safety_agent.config import DEFAULT_PHASE9_PRODUCT_DEMOS_PATH, DEFAULT_PHASE9_SHOP_DEMOS_PATH, PROJECT_ROOT
from trust_safety_agent.demo_cases import load_demo_products, load_demo_shops


def test_phase9_demo_case_counts_and_assets() -> None:
    shops = load_demo_shops(DEFAULT_PHASE9_SHOP_DEMOS_PATH)
    products = load_demo_products(DEFAULT_PHASE9_PRODUCT_DEMOS_PATH)
    assert len(shops) == 10
    assert len(products) == 18
    assert len({case.shop_id for case, _, _ in shops}) == 10
    assert len({case.product_id for case, _, _ in products}) == 18
    product_scenarios = {scenario for _, _, scenario in products}
    assert {"counterfeit_plus_mba", "mba_plus_tmi", "compatibility_exemption", "blurred_image_manual_review"} <= product_scenarios
    existing_product_assets = [PROJECT_ROOT / case.product_images[0] for case, _, _ in products if "missing" not in case.product_images[0]]
    assert all(path.exists() for path in existing_product_assets)

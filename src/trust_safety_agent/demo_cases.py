"""Load and seed the Phase 9 synthetic Case Store without touching Golden v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from trust_safety_agent.case_store import CaseStore, ProductCase, ShopCase
from trust_safety_agent.phase9_workflow import ProductRuntimeEvidence, ShopRuntimeEvidence


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_demo_shops(path: Path) -> List[Tuple[ShopCase, ShopRuntimeEvidence, str]]:
    return [(ShopCase.model_validate(item["case"]), ShopRuntimeEvidence.model_validate(item["runtime_evidence"]), item["scenario"]) for item in _records(path)]


def load_demo_products(path: Path) -> List[Tuple[ProductCase, ProductRuntimeEvidence, str]]:
    return [(ProductCase.model_validate(item["case"]), ProductRuntimeEvidence.model_validate(item["runtime_evidence"]), item["scenario"]) for item in _records(path)]


def seed_case_store(store: CaseStore, shop_path: Path, product_path: Path) -> tuple[int, int]:
    shops = 0
    products = 0
    for case, _, _ in load_demo_shops(shop_path):
        if store.get_shop(case.shop_id) is None:
            store.create_shop(case)
            shops += 1
    for case, _, _ in load_demo_products(product_path):
        if store.get_product(case.product_id) is None:
            store.create_product(case)
            products += 1
    return shops, products

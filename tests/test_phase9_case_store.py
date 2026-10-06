import json
from pathlib import Path

import pytest

from trust_safety_agent.case_store import (
    AgentRunRecord,
    CaseStore,
    JsonlStore,
    LabelStatus,
    ProductCase,
    ReviewerAnnotationRecord,
    RunDecision,
    ShopCase,
    export_records,
    import_product_cases,
    import_shop_cases,
    read_tabular_rows,
)


def test_case_store_revisions_archive_and_raw_run_separation(tmp_path: Path) -> None:
    cases = CaseStore(tmp_path / "cases")
    shop = cases.create_shop(ShopCase(shop_id="S-1", shop_name="Snow Handmade", shop_avatar="data/demo_assets/v5/CASE-V5-011.png", brand="Snow", brand_authorized=None))
    product = cases.create_product(ProductCase(product_id="P-1", title="Nike shoe", product_images=["data/demo_assets/v5/CASE-V5-007.png"], brand="Nike", brand_authorized=False))
    assert shop.brand_authorized is None
    assert product.revision == 1
    assert cases.revise_product("P-1", {"description": "updated"}).revision == 2
    assert cases.archive_shop("S-1").archived is True
    assert cases.list_shops() == []
    assert len(cases.list_shops(include_archived=True)) == 1

    run_store = JsonlStore(tmp_path / "runs.jsonl", AgentRunRecord)
    run = AgentRunRecord(case_id="P-1", case_revision=2, workflow_version="v9", prompt_version="p1", skill_version="s1", model_id="m1", policy_index_version="pv", brand_index_version="bv", suggested_labels={"MBA": LabelStatus.HIT}, suggested_decision=RunDecision.REJECT, confidence=0.9, trace_id="TRC-000000000001")
    run_store.append(run)
    review_store = JsonlStore(tmp_path / "reviews.jsonl", ReviewerAnnotationRecord)
    review_store.append(ReviewerAnnotationRecord(run_id=run.run_id, reviewer_labels={"MBA": LabelStatus.NOT_HIT}, reviewer_decision=RunDecision.APPROVE, reviewer_note="authorization supplied"))
    assert run_store.list()[0].suggested_decision == RunDecision.REJECT
    assert review_store.list()[0].reviewer_decision == RunDecision.APPROVE


def test_case_export_csv_and_excel(tmp_path: Path) -> None:
    records = [ProductCase(product_id="P-2", title="Case", product_images=["a.png"], brand_authorized=None)]
    csv_path = tmp_path / "products.csv"
    xlsx_path = tmp_path / "products.xlsx"
    export_records(records, csv_path)
    export_records(records, xlsx_path)
    assert json.loads(read_tabular_rows(csv_path)[0]["product_images"]) == ["a.png"]
    assert read_tabular_rows(xlsx_path)[0]["product_id"] == "P-2"
    with pytest.raises(ValueError):
        export_records([], tmp_path / "empty.csv")


def test_case_import_validates_nullable_authorization_duplicates_and_bad_urls(tmp_path: Path) -> None:
    product_csv = tmp_path / "products.csv"
    product_csv.write_text("product_id,title,product_images,brand_authorized\nP-10,Phone case,images/case.png,null\n", encoding="utf-8")
    store = CaseStore(tmp_path / "store")
    imported = import_product_cases(product_csv, store)
    assert imported[0].brand_authorized is None
    assert imported[0].product_images == ["images/case.png"]

    shop_csv = tmp_path / "shops.csv"
    shop_csv.write_text("shop_id,shop_name,shop_avatar,brand_authorized\nS-1,Store,ftp://bad.example/avatar.png,false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="media reference"):
        import_shop_cases(shop_csv, store)

    duplicate_csv = tmp_path / "duplicate.csv"
    duplicate_csv.write_text("product_id,title,product_images\nP-11,A,a.png\nP-11,B,b.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate product_id"):
        import_product_cases(duplicate_csv, store)

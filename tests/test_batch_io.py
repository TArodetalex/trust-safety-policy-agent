import io

import pytest
from openpyxl import Workbook

from trust_safety_agent.batch_io import (
    BatchValidationError,
    export_csv,
    export_xlsx,
    parse_product_batch,
    parse_shop_batch,
)
from trust_safety_agent.product_review import ProductReviewDecision, ProductReviewResult, ReviewConfidence


def _xlsx_bytes(headers: list[str], rows: list[list[str]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_parse_shop_csv_and_excel() -> None:
    csv_payload = (
        "case_id,shop_name,avatar_url,controlled_brand,authorization_status\n"
        "SHOP-1,Gucci Official,,Gucci,unknown\n"
    ).encode()
    csv_records = parse_shop_batch(csv_payload, "shops.csv")
    xlsx_records = parse_shop_batch(
        _xlsx_bytes(
            ["case_id", "shop_name", "avatar_url", "controlled_brand"],
            [["SHOP-2", "Nike Outlet", "https://example.com/a.png", "Nike"]],
        ),
        "shops.xlsx",
    )

    assert csv_records[0].case_id == "SHOP-1"
    assert xlsx_records[0].case_id == "SHOP-2"


def test_batch_rejects_empty_missing_duplicate_and_invalid_url() -> None:
    with pytest.raises(BatchValidationError, match="文件为空"):
        parse_shop_batch(b"", "shops.csv")
    with pytest.raises(BatchValidationError, match="缺少必填列"):
        parse_shop_batch(b"case_id,shop_name\nA,Store\n", "shops.csv")
    duplicate = (
        "case_id,shop_name,avatar_url,controlled_brand\n"
        "A,Store 1,,Gucci\nA,Store 2,,Nike\n"
    ).encode()
    with pytest.raises(BatchValidationError, match="重复"):
        parse_shop_batch(duplicate, "shops.csv")
    invalid_url = (
        "case_id,shop_name,avatar_url,controlled_brand\n"
        "A,Store,not-a-url,Gucci\n"
    ).encode()
    with pytest.raises(BatchValidationError, match="url"):
        parse_shop_batch(invalid_url, "shops.csv")


def test_parse_product_batch_and_export_formats() -> None:
    payload = (
        "case_id,product_id,title,description,image_url,optional_brand_field\n"
        "PR-10,SKU-10,Gucci bag,replica,,Gucci\n"
    ).encode()
    records = parse_product_batch(payload, "products.csv")
    result = ProductReviewResult(
        case_id=records[0].case_id,
        suggested_decision=ProductReviewDecision.MANUAL_REVIEW,
        confidence=ReviewConfidence.LOW,
        reason="review",
    )

    assert records[0].optional_brand_field == "Gucci"
    assert export_csv([result]).startswith(b"\xef\xbb\xbf")
    assert export_xlsx([result]).startswith(b"PK")

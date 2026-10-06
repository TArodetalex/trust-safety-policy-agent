"""Validated Phase 8 synthetic demo cases and asset integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import List, Optional

from pydantic import Field, model_validator

from trust_safety_agent.product_review import (
    BrandAuthorizationStatus,
    DesignSimilaritySignal,
    ImageQuality,
    LogoMatchType,
    ProductReviewDecision,
    ProductRiskSubtype,
)
from trust_safety_agent.schema import StrictModel, StringEnum


class DemoWorkflow(StringEnum):
    PRODUCT_IPR = "product_ipr"
    SHOP_IDENTITY = "shop_identity"


class SyntheticDemoCase(StrictModel):
    schema_version: str = "1.0.0"
    dataset_version: str = "v5-candidate-1.0.0"
    case_id: str = Field(pattern=r"^CASE-V5-\d{3}$")
    name: str = Field(min_length=2, max_length=100)
    workflow: DemoWorkflow
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=4000)
    image_path: str = Field(max_length=500)
    optional_brand_field: Optional[str] = Field(default=None, max_length=100)
    observed_product_brand: Optional[str] = Field(default=None, max_length=100)
    ocr_text: str = Field(default="", max_length=4000)
    visual_marks: List[str] = Field(default_factory=list, max_length=30)
    brand_authorization_status: BrandAuthorizationStatus = BrandAuthorizationStatus.UNKNOWN
    logo_match_type: LogoMatchType = LogoMatchType.NOT_ASSESSED
    design_similarity: DesignSimilaritySignal = DesignSimilaritySignal.NOT_ASSESSED
    independent_brand_registered: bool = False
    listing_price: Optional[float] = Field(default=None, gt=0)
    reference_price: Optional[float] = Field(default=None, gt=0)
    image_quality: ImageQuality = ImageQuality.NOT_ASSESSED
    visual_confidence: float = Field(default=1.0, ge=0, le=1)
    shop_name: Optional[str] = Field(default=None, max_length=300)
    controlled_brand: Optional[str] = Field(default=None, max_length=100)
    expected_decision: ProductReviewDecision
    expected_policy_id: Optional[str] = None
    expected_risk_subtype: Optional[ProductRiskSubtype] = None
    human_reason: str = Field(min_length=10, max_length=1000)
    source: str = "synthetic_phase8"
    split: str = "candidate"

    @model_validator(mode="after")
    def validate_workflow_fields(self) -> "SyntheticDemoCase":
        path = PurePosixPath(self.image_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path must be a safe project-relative path")
        if self.workflow == DemoWorkflow.SHOP_IDENTITY and not (
            self.shop_name and self.controlled_brand
        ):
            raise ValueError("shop cases require shop_name and controlled_brand")
        return self


class SyntheticAssetRecord(StrictModel):
    case_image: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0)
    generation_method: str = "OpenAI built-in image generation"
    contains_real_brand: bool = False


class SyntheticAssetManifest(StrictModel):
    dataset_version: str = "v5-candidate-1.0.0"
    generated_asset_count: int
    records: List[SyntheticAssetRecord]


def load_synthetic_demo_cases(path: Path) -> List[SyntheticDemoCase]:
    records = [
        SyntheticDemoCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 12 <= len(records) <= 20:
        raise ValueError("Phase 8 demo dataset must contain 12 to 20 cases")
    if len({item.case_id for item in records}) != len(records):
        raise ValueError("Phase 8 demo case IDs must be unique")
    return records


def build_asset_manifest(project_root: Path, cases: List[SyntheticDemoCase]) -> SyntheticAssetManifest:
    paths = sorted({item.image_path for item in cases})
    records = []
    for relative in paths:
        path = project_root / relative
        payload = path.read_bytes()
        records.append(
            SyntheticAssetRecord(
                case_image=relative,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            )
        )
    return SyntheticAssetManifest(
        generated_asset_count=len(records),
        records=records,
    )


def verify_asset_manifest(project_root: Path, manifest_path: Path) -> List[str]:
    manifest = SyntheticAssetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    errors = []
    for record in manifest.records:
        path = project_root / record.case_image
        if not path.exists():
            errors.append(f"missing asset: {record.case_image}")
            continue
        payload = path.read_bytes()
        if len(payload) != record.bytes:
            errors.append(f"size mismatch: {record.case_image}")
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            errors.append(f"sha256 mismatch: {record.case_image}")
    return errors

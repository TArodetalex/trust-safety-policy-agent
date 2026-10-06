"""Deterministic Product IPR review pipeline with auditable evidence."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Dict, List, Optional

from pydantic import Field, HttpUrl, model_validator

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.schema import ExemptionType, StrictModel, StringEnum
from trust_safety_agent.vector_store import PolicyVectorStore


class ProductReviewDecision(StringEnum):
    APPROVE = "approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


class ReviewConfidence(StringEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BrandAuthorizationStatus(StringEnum):
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"


class LogoMatchType(StringEnum):
    EXACT = "exact"
    MODIFIED = "modified"
    NONE = "none"
    UNCLEAR = "unclear"
    NOT_ASSESSED = "not_assessed"


class DesignSimilaritySignal(StringEnum):
    DISTINCTIVE_COPY = "distinctive_copy"
    COMMON_STYLE = "common_style"
    NONE = "none"
    UNCLEAR = "unclear"
    NOT_ASSESSED = "not_assessed"


class ImageQuality(StringEnum):
    CLEAR = "clear"
    UNCLEAR = "unclear"
    NOT_ASSESSED = "not_assessed"


class ProductRiskSubtype(StringEnum):
    COUNTERFEIT = "counterfeit"
    KNOCKOFF_MODIFIED_LOGO = "knockoff_modified_logo"
    KNOCKOFF_DISTINCTIVE_DESIGN = "knockoff_distinctive_design"
    KNOCKOFF_WHITE_LABEL = "knockoff_white_label"
    MISSING_BRAND_AUTHORIZATION = "missing_brand_authorization"
    TEXT_MERCHANDISE_INCONSISTENCY = "text_merchandise_inconsistency"


class CandidateSource(StringEnum):
    BRAND_FIELD = "brand_field"
    TITLE = "title"
    DESCRIPTION = "description"
    OCR = "ocr"
    IMAGE_VISUAL = "image_visual"


class ProductReviewInput(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    product_id: Optional[str] = Field(default=None, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=4000)
    optional_brand_field: Optional[str] = Field(default=None, max_length=100)
    optional_category: Optional[str] = Field(default=None, max_length=100)
    image_url: Optional[HttpUrl] = None
    image_path: Optional[str] = Field(default=None, max_length=500)
    ocr_text: str = Field(default="", max_length=4000)
    visual_marks: List[str] = Field(default_factory=list, max_length=30)
    brand_authorization_status: BrandAuthorizationStatus = (
        BrandAuthorizationStatus.UNKNOWN
    )
    observed_product_brand: Optional[str] = Field(default=None, max_length=100)
    logo_match_type: LogoMatchType = LogoMatchType.NOT_ASSESSED
    design_similarity: DesignSimilaritySignal = DesignSimilaritySignal.NOT_ASSESSED
    independent_brand_registered: bool = False
    listing_price: Optional[float] = Field(default=None, gt=0)
    reference_price: Optional[float] = Field(default=None, gt=0)
    image_quality: ImageQuality = ImageQuality.NOT_ASSESSED
    visual_confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_image_source(self) -> "ProductReviewInput":
        if self.image_url and self.image_path:
            raise ValueError("product cannot declare both image_url and image_path")
        if self.image_path:
            path = PurePosixPath(self.image_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("image_path must be a safe project-relative path")
        return self


class BrandCandidate(StrictModel):
    candidate_brand: str
    source: CandidateSource
    evidence: str
    controlled_brand_match: bool
    matched_brand: Optional[str] = None
    context_confidence: ReviewConfidence
    ambiguity: bool = False
    notes: str = ""


class ProductEvidence(StrictModel):
    source: CandidateSource
    text: str
    signal: str


class ProductPolicyReference(StrictModel):
    policy_id: str
    chunk_id: str
    title: str
    quote: str
    retrieval_score: float = Field(ge=0, le=1)


class ProductReviewResult(StrictModel):
    case_id: str
    candidate_brands: List[BrandCandidate] = Field(default_factory=list)
    controlled_brand_matches: List[str] = Field(default_factory=list)
    evidence: List[ProductEvidence] = Field(default_factory=list)
    policy_references: List[ProductPolicyReference] = Field(default_factory=list)
    possible_exemptions: List[ExemptionType] = Field(default_factory=list)
    suggested_decision: ProductReviewDecision
    confidence: ReviewConfidence
    reviewer_checkpoints: List[str] = Field(default_factory=list)
    risk_subtype: Optional[ProductRiskSubtype] = None
    reason: str


_COUNTERFEIT = re.compile(
    r"\b(counterfeit|fake|replica|mirror\s*copy|aaa\s*copy)\b|1\s*:\s*1|"
    r"高仿|假货|仿品|复刻",
    re.IGNORECASE,
)
_COMPATIBILITY = re.compile(
    r"\b(compatible\s+with|fits?|for\s+use\s+with)\b|适用于|兼容",
    re.IGNORECASE,
)
_SECOND_HAND = re.compile(
    r"\b(pre[- ]?owned|used|second[- ]?hand)\b|二手|中古",
    re.IGNORECASE,
)
_BRAND_CONTEXT = re.compile(
    r"\b(brand|logo|official|genuine|authentic|store|shop)\b|"
    r"品牌|标识|官方|正品|旗舰店",
    re.IGNORECASE,
)


class ProductReviewAssistant:
    def __init__(
        self,
        library: ControlledBrandLibrary,
        store: PolicyVectorStore,
        top_k: int = 5,
    ) -> None:
        self.library = library
        self.store = store
        self.top_k = top_k

    def _candidate_sources(
        self, item: ProductReviewInput
    ) -> List[tuple[CandidateSource, str]]:
        values = [
            (CandidateSource.BRAND_FIELD, item.optional_brand_field or ""),
            (CandidateSource.TITLE, item.title),
            (CandidateSource.DESCRIPTION, item.description),
            (CandidateSource.OCR, item.ocr_text),
            (CandidateSource.IMAGE_VISUAL, " ".join(item.visual_marks)),
        ]
        return [(source, text) for source, text in values if text.strip()]

    def _recall_candidates(
        self, item: ProductReviewInput
    ) -> List[BrandCandidate]:
        candidates: List[BrandCandidate] = []
        seen: set[tuple[str, CandidateSource]] = set()
        for source, text in self._candidate_sources(item):
            mentions = self.library.find_mentions(text)
            for mention in mentions:
                key = (mention.brand_name, source)
                if key in seen:
                    continue
                seen.add(key)
                contextual = bool(
                    _BRAND_CONTEXT.search(text)
                    or _COUNTERFEIT.search(text)
                    or _COMPATIBILITY.search(text)
                    or _SECOND_HAND.search(text)
                )
                ambiguous = mention.is_common_word and not (
                    source in {CandidateSource.BRAND_FIELD, CandidateSource.IMAGE_VISUAL}
                    or contextual
                )
                candidates.append(
                    BrandCandidate(
                        candidate_brand=mention.matched_alias,
                        source=source,
                        evidence=mention.evidence,
                        controlled_brand_match=True,
                        matched_brand=mention.brand_name,
                        context_confidence=(
                            ReviewConfidence.LOW
                            if ambiguous
                            else ReviewConfidence.HIGH
                        ),
                        ambiguity=ambiguous,
                        notes=(
                            "普通词或短词品牌，需核对是否为商标语境。"
                            if ambiguous
                            else "已命中受控品牌库。"
                        ),
                    )
                )

        brand_field = (item.optional_brand_field or "").strip()
        if brand_field and not any(
            candidate.source == CandidateSource.BRAND_FIELD for candidate in candidates
        ):
            candidates.insert(
                0,
                BrandCandidate(
                    candidate_brand=brand_field,
                    source=CandidateSource.BRAND_FIELD,
                    evidence=brand_field,
                    controlled_brand_match=False,
                    context_confidence=ReviewConfidence.LOW,
                    ambiguity=True,
                    notes="未命中受控品牌库，需人工确认品牌归属。",
                ),
            )
        return candidates

    def _policy_references(
        self, query: str, policy_ids: set[str]
    ) -> List[ProductPolicyReference]:
        hits = self.store.retrieve(
            query,
            top_k=max(self.top_k, min(self.store.count(), 20)),
        )
        references = []
        for hit in hits:
            if hit.chunk.policy_id not in policy_ids:
                continue
            references.append(
                ProductPolicyReference(
                    policy_id=hit.chunk.policy_id,
                    chunk_id=hit.chunk.chunk_id,
                    title=hit.chunk.title,
                    quote=hit.chunk.content,
                    retrieval_score=hit.score,
                )
            )
        return references

    def review(self, item: ProductReviewInput) -> ProductReviewResult:
        candidates = self._recall_candidates(item)
        combined = " ".join(text for _, text in self._candidate_sources(item))
        controlled = sorted(
            {
                candidate.matched_brand
                for candidate in candidates
                if candidate.matched_brand
            }
        )
        has_counterfeit = bool(_COUNTERFEIT.search(combined))
        has_compatibility = bool(_COMPATIBILITY.search(combined))
        has_second_hand = bool(_SECOND_HAND.search(combined))
        ambiguous = any(candidate.ambiguity for candidate in candidates)
        has_unmatched_brand = any(
            not candidate.controlled_brand_match for candidate in candidates
        )
        image_without_evidence = bool(item.image_url or item.image_path) and not (
            item.ocr_text.strip()
            or item.visual_marks
            or item.observed_product_brand
            or item.logo_match_type != LogoMatchType.NOT_ASSESSED
            or item.design_similarity != DesignSimilaritySignal.NOT_ASSESSED
        )

        information_text = " ".join(
            value
            for value in [item.optional_brand_field or "", item.title, item.description]
            if value.strip()
        )
        merchandise_text = " ".join(
            value
            for value in [
                item.observed_product_brand or "",
                item.ocr_text,
                " ".join(item.visual_marks),
            ]
            if value.strip()
        )
        information_brands = {
            mention.brand_name for mention in self.library.find_mentions(information_text)
        }
        merchandise_brands = {
            mention.brand_name for mention in self.library.find_mentions(merchandise_text)
        }
        observed_normalized = (item.observed_product_brand or "").strip().casefold()
        white_label = observed_normalized in {
            "unbranded",
            "white label",
            "none",
            "白牌",
            "无品牌",
        }
        text_merchandise_mismatch = bool(information_brands) and (
            white_label
            or bool(
                merchandise_brands
                and information_brands.isdisjoint(merchandise_brands)
            )
        )
        suspicious_price = bool(
            item.listing_price
            and item.reference_price
            and item.listing_price / item.reference_price <= 0.3
        )
        visual_signal_present = (
            item.logo_match_type
            in {LogoMatchType.EXACT, LogoMatchType.MODIFIED, LogoMatchType.UNCLEAR}
            or item.design_similarity
            in {
                DesignSimilaritySignal.DISTINCTIVE_COPY,
                DesignSimilaritySignal.UNCLEAR,
            }
        )
        visual_uncertain = visual_signal_present and (
            item.image_quality == ImageQuality.UNCLEAR
            or item.visual_confidence < 0.7
            or item.logo_match_type == LogoMatchType.UNCLEAR
            or item.design_similarity == DesignSimilaritySignal.UNCLEAR
        )

        evidence: List[ProductEvidence] = []
        if has_counterfeit:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.DESCRIPTION,
                    text=combined,
                    signal="explicit_counterfeit_language",
                )
            )
        if has_compatibility:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.DESCRIPTION,
                    text=combined,
                    signal="compatibility_context",
                )
            )
        if has_second_hand:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.DESCRIPTION,
                    text=combined,
                    signal="second_hand_context",
                )
            )
        if suspicious_price:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.DESCRIPTION,
                    text=(
                        f"listing_price={item.listing_price}; "
                        f"reference_price={item.reference_price}"
                    ),
                    signal="material_price_gap_supporting_signal",
                )
            )
        if item.logo_match_type != LogoMatchType.NOT_ASSESSED:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.IMAGE_VISUAL,
                    text=item.logo_match_type.value,
                    signal="logo_match_type",
                )
            )
        if item.design_similarity != DesignSimilaritySignal.NOT_ASSESSED:
            evidence.append(
                ProductEvidence(
                    source=CandidateSource.IMAGE_VISUAL,
                    text=item.design_similarity.value,
                    signal="design_similarity",
                )
            )

        checkpoints: List[str] = []
        exemptions: List[ExemptionType] = []
        if has_compatibility:
            exemptions.append(ExemptionType.COMPATIBILITY)
        if has_second_hand:
            exemptions.append(ExemptionType.SECOND_HAND)

        decision = ProductReviewDecision.APPROVE
        confidence = ReviewConfidence.MEDIUM
        reason = "未发现可归因的知识产权违规信号。"
        risk_subtype: Optional[ProductRiskSubtype] = None
        target_policy_ids: set[str] = set()

        if visual_uncertain:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.LOW
            reason = "图片质量或视觉置信度不足，无法可靠区分一致 Logo、变形 Logo 与外观相似。"
            checkpoints.append("查看清晰原图，确认 Logo 关系和具有识别性的外观元素。")
        elif has_counterfeit and (
            item.logo_match_type == LogoMatchType.EXACT or controlled
        ):
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            risk_subtype = ProductRiskSubtype.COUNTERFEIT
            target_policy_ids = {"POL-CF-001"}
            reason = "完整品牌指示与明确假货/复制品表述共现；低价仅作为辅助风险信号。"
            checkpoints.append("确认商品文案中的非正品表述与对应品牌直接相关。")
        elif item.logo_match_type == LogoMatchType.MODIFIED:
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            risk_subtype = ProductRiskSubtype.KNOCKOFF_MODIFIED_LOGO
            target_policy_ids = {"POL-KO-001"}
            reason = "商品使用了对现有商标的可识别变形或魔改 Logo。"
        elif item.design_similarity == DesignSimilaritySignal.DISTINCTIVE_COPY:
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            risk_subtype = (
                ProductRiskSubtype.KNOCKOFF_WHITE_LABEL
                if white_label
                else ProductRiskSubtype.KNOCKOFF_DISTINCTIVE_DESIGN
            )
            target_policy_ids = {"POL-KO-001"}
            reason = "商品复制了具有识别性的非通用外观元素；普通风格相似不足以触发该规则。"
        elif (
            item.brand_authorization_status
            == BrandAuthorizationStatus.UNAUTHORIZED
            and (information_brands or merchandise_brands)
            and not has_compatibility
        ):
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            risk_subtype = ProductRiskSubtype.MISSING_BRAND_AUTHORIZATION
            target_policy_ids = {"POL-MBA-001"}
            reason = "品牌授权状态为未授权，且文本或商品实物存在完整品牌指示。"
        elif text_merchandise_mismatch and not has_compatibility:
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            risk_subtype = ProductRiskSubtype.TEXT_MERCHANDISE_INCONSISTENCY
            target_policy_ids = {"POL-TMI-001"}
            reason = "信息层指向的品牌与商品实物品牌不一致，或实物明确为白牌。"
        elif has_compatibility or has_second_hand:
            decision = ProductReviewDecision.APPROVE
            confidence = ReviewConfidence.HIGH
            reason = "品牌使用处于明确的兼容性或二手转售语境，且未发现独立违规信号。"
            checkpoints.append("确认文案没有暗示品牌授权、赞助或官方生产。")
            target_policy_ids = {"POL-EX-001"}
        elif (
            item.brand_authorization_status == BrandAuthorizationStatus.AUTHORIZED
            and not has_counterfeit
        ):
            decision = ProductReviewDecision.APPROVE
            confidence = ReviewConfidence.HIGH
            reason = "品牌授权已确认，且信息层与商品实物未出现冲突或独立违规信号。"
            target_policy_ids = {"POL-EX-001"}
        elif item.independent_brand_registered and not (
            item.logo_match_type == LogoMatchType.EXACT
            or item.design_similarity == DesignSimilaritySignal.DISTINCTIVE_COPY
        ):
            decision = ProductReviewDecision.APPROVE
            confidence = ReviewConfidence.HIGH
            reason = "独立品牌注册信息已提供，且未发现完整 Logo 复制或独特外观复制。"
            target_policy_ids = {"POL-EX-001"}
        elif image_without_evidence:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.LOW
            reason = "已提供商品图片，但当前没有可靠的 OCR 或视觉标记，无法自动确认品牌与侵权语境。"
            checkpoints.append("人工检查图片中的 Logo、品牌字样、包装和授权信息。")
        elif has_unmatched_brand or ambiguous:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.LOW
            reason = "品牌候选存在词义歧义或未收录品牌，暂不适合自动决策。"
            checkpoints.append("确认候选词在当前商品中是否代表受保护品牌。")
        elif controlled:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.MEDIUM
            reason = "已命中受控品牌，但品牌命中本身不足以证明侵权或仿冒。"
            checkpoints.append("核对商品真伪、授权资料与品牌使用方式。")
        references = self._policy_references(
            f"{combined} {risk_subtype.value if risk_subtype else ''}",
            target_policy_ids,
        )
        if decision == ProductReviewDecision.REJECT and not references:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.LOW
            reason = "检测到高风险组合，但未召回可直接支撑拒绝的策略证据。"
            checkpoints.append("补充或核对仿冒商品策略证据。")

        return ProductReviewResult(
            case_id=item.case_id,
            candidate_brands=candidates,
            controlled_brand_matches=controlled,
            evidence=evidence,
            policy_references=references,
            possible_exemptions=exemptions,
            suggested_decision=decision,
            confidence=confidence,
            reviewer_checkpoints=checkpoints,
            risk_subtype=risk_subtype,
            reason=reason,
        )

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

    def _policy_references(self, query: str, reject: bool) -> List[ProductPolicyReference]:
        hits = self.store.retrieve(query, top_k=self.top_k)
        references = []
        for hit in hits:
            is_counterfeit = hit.chunk.policy_id == "POL-CF-001"
            is_exemption = hit.chunk.policy_id == "POL-EX-001"
            if (reject and not is_counterfeit) or (not reject and not is_exemption):
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
            item.ocr_text.strip() or item.visual_marks
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

        checkpoints: List[str] = []
        exemptions: List[ExemptionType] = []
        if has_compatibility:
            exemptions.append(ExemptionType.COMPATIBILITY)
        if has_second_hand:
            exemptions.append(ExemptionType.SECOND_HAND)

        reject = bool(controlled and has_counterfeit)
        if reject:
            decision = ProductReviewDecision.REJECT
            confidence = ReviewConfidence.HIGH
            reason = "受控品牌与明确假货/复制品表述共现，符合仿冒商品策略的拒绝条件。"
            checkpoints.append("确认商品文案中的非正品表述与对应品牌直接相关。")
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
        elif controlled and (has_compatibility or has_second_hand):
            decision = ProductReviewDecision.APPROVE
            confidence = ReviewConfidence.HIGH
            reason = "品牌使用处于明确的兼容性或二手转售语境，且未发现独立违规信号。"
            checkpoints.append("确认文案没有暗示品牌授权、赞助或官方生产。")
        elif controlled:
            decision = ProductReviewDecision.MANUAL_REVIEW
            confidence = ReviewConfidence.MEDIUM
            reason = "已命中受控品牌，但品牌命中本身不足以证明侵权或仿冒。"
            checkpoints.append("核对商品真伪、授权资料与品牌使用方式。")
        else:
            decision = ProductReviewDecision.APPROVE
            confidence = ReviewConfidence.MEDIUM
            reason = "未召回受控品牌，也未发现可归因的知识产权违规信号。"

        references = self._policy_references(combined, reject=reject)
        if reject and not references:
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
            reason=reason,
        )

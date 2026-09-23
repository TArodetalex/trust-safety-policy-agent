"""Auditable shop-name and avatar identity review with authorization gating."""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import Field, HttpUrl

from trust_safety_agent.brand_library import (
    ControlledBrand,
    ControlledBrandLibrary,
    normalize_brand_text,
)
from trust_safety_agent.product_review import (
    ProductPolicyReference,
    ProductReviewDecision,
    ReviewConfidence,
)
from trust_safety_agent.schema import StrictModel, StringEnum
from trust_safety_agent.vector_store import PolicyVectorStore


class AuthorizationStatus(StringEnum):
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"


class ShopNameSignalType(StringEnum):
    EXACT_BRAND = "exact_brand"
    MODIFIED_BRAND = "modified_brand"
    IMPERSONATION = "impersonation"
    MEANINGFUL_COMMON_WORD = "meaningful_common_word"
    SUBSTRING_COINCIDENCE = "substring_coincidence"
    UNRELATED_USE = "unrelated_use"
    WEAK_EVIDENCE = "weak_evidence"


class AvatarSignalType(StringEnum):
    BRAND_LOGO = "brand_logo"
    MODIFIED_LOGO = "modified_logo"
    CLEAR_BRAND_VISUAL = "clear_brand_visual"
    INCIDENTAL_EXPOSURE = "incidental_exposure"
    UNRELATED_OBJECT = "unrelated_object"
    WEAK_EVIDENCE = "weak_evidence"
    UNAVAILABLE_IMAGE = "unavailable_image"


class ShopIdentityInput(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    shop_name: str = Field(min_length=1, max_length=300)
    avatar_url: Optional[HttpUrl] = None
    controlled_brand: str = Field(min_length=1, max_length=100)
    authorization_status: AuthorizationStatus = AuthorizationStatus.UNKNOWN
    avatar_visual_marks: List[str] = Field(default_factory=list, max_length=30)


class ShopNameSignal(StrictModel):
    signal_type: ShopNameSignalType
    matched_text: str = ""
    evidence: str
    confidence: ReviewConfidence


class AvatarSignal(StrictModel):
    signal_type: AvatarSignalType
    evidence: str
    confidence: ReviewConfidence


class ShopIdentityResult(StrictModel):
    case_id: str
    controlled_brand: str
    shop_name_signal: ShopNameSignal
    avatar_signal: AvatarSignal
    authorization: AuthorizationStatus
    policy_references: List[ProductPolicyReference] = Field(default_factory=list)
    possible_exemptions: List[str] = Field(default_factory=list)
    suggested_decision: ProductReviewDecision
    confidence: ReviewConfidence
    reason: str
    reviewer_checkpoints: List[str] = Field(default_factory=list)


_IMPERSONATION = re.compile(
    r"\b(official|flagship|authorized|verified|outlet)\b|"
    r"官方|旗舰|授权|认证|专卖店",
    re.IGNORECASE,
)
_COMMON_CONTEXT = re.compile(
    r"\b(apple\s+juice|fruit|coach(?:ing)?|bus|training)\b|"
    r"苹果汁|水果|教练|培训|巴士",
    re.IGNORECASE,
)


def _aliases(brand: ControlledBrand) -> List[str]:
    return [brand.brand_name, *brand.aliases]


class ShopIdentityReviewer:
    def __init__(
        self,
        library: ControlledBrandLibrary,
        store: PolicyVectorStore,
        top_k: int = 5,
    ) -> None:
        self.library = library
        self.store = store
        self.top_k = top_k

    def _brand(self, value: str) -> Optional[ControlledBrand]:
        return self.library.get(value)

    def judge_shop_name(
        self,
        shop_name: str,
        brand: Optional[ControlledBrand],
    ) -> ShopNameSignal:
        if brand is None:
            return ShopNameSignal(
                signal_type=ShopNameSignalType.WEAK_EVIDENCE,
                evidence=shop_name,
                confidence=ReviewConfidence.LOW,
            )

        normalized = normalize_brand_text(shop_name)
        normalized_aliases = [normalize_brand_text(alias) for alias in _aliases(brand)]
        exact = normalized in normalized_aliases
        mentions = self.library.find_mentions(shop_name)
        brand_mentioned = any(item.brand_name == brand.brand_name for item in mentions)
        if brand.is_common_word and _COMMON_CONTEXT.search(shop_name):
            return ShopNameSignal(
                signal_type=ShopNameSignalType.MEANINGFUL_COMMON_WORD,
                matched_text=brand.brand_name,
                evidence=shop_name,
                confidence=ReviewConfidence.HIGH,
            )
        if (exact or brand_mentioned) and _IMPERSONATION.search(shop_name):
            return ShopNameSignal(
                signal_type=ShopNameSignalType.IMPERSONATION,
                matched_text=brand.brand_name,
                evidence=shop_name,
                confidence=ReviewConfidence.HIGH,
            )
        if exact:
            return ShopNameSignal(
                signal_type=ShopNameSignalType.EXACT_BRAND,
                matched_text=brand.brand_name,
                evidence=shop_name,
                confidence=ReviewConfidence.HIGH,
            )
        if brand_mentioned:
            return ShopNameSignal(
                signal_type=ShopNameSignalType.MODIFIED_BRAND,
                matched_text=brand.brand_name,
                evidence=shop_name,
                confidence=ReviewConfidence.MEDIUM,
            )
        compact_name = re.sub(r"[^\w]", "", normalized)
        if any(
            re.sub(r"[^\w]", "", alias) in compact_name
            for alias in normalized_aliases
        ):
            return ShopNameSignal(
                signal_type=ShopNameSignalType.SUBSTRING_COINCIDENCE,
                matched_text=brand.brand_name,
                evidence=shop_name,
                confidence=ReviewConfidence.LOW,
            )
        return ShopNameSignal(
            signal_type=ShopNameSignalType.UNRELATED_USE,
            evidence=shop_name,
            confidence=ReviewConfidence.MEDIUM,
        )

    def judge_avatar(
        self,
        item: ShopIdentityInput,
        brand: Optional[ControlledBrand],
    ) -> AvatarSignal:
        marks = " ".join(item.avatar_visual_marks).strip()
        if not marks:
            return AvatarSignal(
                signal_type=AvatarSignalType.UNAVAILABLE_IMAGE,
                evidence=str(item.avatar_url or "未提供头像"),
                confidence=ReviewConfidence.LOW,
            )
        if re.search(r"incidental|background|背景|偶然", marks, re.IGNORECASE):
            return AvatarSignal(
                signal_type=AvatarSignalType.INCIDENTAL_EXPOSURE,
                evidence=marks,
                confidence=ReviewConfidence.MEDIUM,
            )
        if brand and any(
            normalize_brand_text(alias) in normalize_brand_text(marks)
            for alias in _aliases(brand)
        ):
            signal = (
                AvatarSignalType.BRAND_LOGO
                if re.search(r"logo|标识|商标", marks, re.IGNORECASE)
                else AvatarSignalType.CLEAR_BRAND_VISUAL
            )
            return AvatarSignal(
                signal_type=signal,
                evidence=marks,
                confidence=ReviewConfidence.HIGH,
            )
        return AvatarSignal(
            signal_type=AvatarSignalType.WEAK_EVIDENCE,
            evidence=marks,
            confidence=ReviewConfidence.LOW,
        )

    def _policy_references(self, query: str) -> List[ProductPolicyReference]:
        references = []
        for hit in self.store.retrieve(query, top_k=self.top_k):
            if hit.chunk.policy_id != "POL-SI-001":
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

    def review(self, item: ShopIdentityInput) -> ShopIdentityResult:
        brand = self._brand(item.controlled_brand)
        name_signal = self.judge_shop_name(item.shop_name, brand)
        avatar_signal = self.judge_avatar(item, brand)

        if item.authorization_status == AuthorizationStatus.AUTHORIZED:
            return ShopIdentityResult(
                case_id=item.case_id,
                controlled_brand=item.controlled_brand,
                shop_name_signal=name_signal,
                avatar_signal=avatar_signal,
                authorization=item.authorization_status,
                possible_exemptions=["documented_authorization"],
                suggested_decision=ProductReviewDecision.APPROVE,
                confidence=ReviewConfidence.HIGH,
                reason="授权状态已明确确认，品牌身份信号不再作为拒绝证据。",
                reviewer_checkpoints=["核对授权资料的有效期、授权主体和适用店铺。"],
            )

        strong_name = name_signal.signal_type == ShopNameSignalType.IMPERSONATION
        strong_avatar = avatar_signal.signal_type in {
            AvatarSignalType.BRAND_LOGO,
            AvatarSignalType.CLEAR_BRAND_VISUAL,
        }
        references = self._policy_references(
            f"{item.controlled_brand} {item.shop_name} official flagship brand logo shop"
        )
        if brand and (strong_name or strong_avatar) and references:
            return ShopIdentityResult(
                case_id=item.case_id,
                controlled_brand=brand.brand_name,
                shop_name_signal=name_signal,
                avatar_signal=avatar_signal,
                authorization=item.authorization_status,
                policy_references=references,
                suggested_decision=ProductReviewDecision.REJECT,
                confidence=ReviewConfidence.HIGH,
                reason="未通过授权 Gate，且店铺名或头像存在明确的品牌官方身份信号。",
                reviewer_checkpoints=["确认该店铺是否会让普通用户误认为品牌官方或授权店铺。"],
            )

        review_required = (
            brand is None
            or name_signal.signal_type
            in {
                ShopNameSignalType.EXACT_BRAND,
                ShopNameSignalType.MODIFIED_BRAND,
                ShopNameSignalType.SUBSTRING_COINCIDENCE,
                ShopNameSignalType.WEAK_EVIDENCE,
            }
            or avatar_signal.signal_type
            in {AvatarSignalType.UNAVAILABLE_IMAGE, AvatarSignalType.WEAK_EVIDENCE}
        )
        if review_required:
            checkpoints = []
            if brand is None:
                checkpoints.append("确认 controlled_brand 是否属于受控品牌库。")
            if item.authorization_status == AuthorizationStatus.UNKNOWN:
                checkpoints.append("核对店铺是否能提供真实且在有效期内的品牌授权。")
            if avatar_signal.signal_type == AvatarSignalType.UNAVAILABLE_IMAGE:
                checkpoints.append("打开头像原图，检查 Logo、变体 Logo 和官方身份标识。")
            return ShopIdentityResult(
                case_id=item.case_id,
                controlled_brand=item.controlled_brand,
                shop_name_signal=name_signal,
                avatar_signal=avatar_signal,
                authorization=item.authorization_status,
                policy_references=references,
                suggested_decision=ProductReviewDecision.MANUAL_REVIEW,
                confidence=ReviewConfidence.LOW,
                reason="授权或店铺身份证据不足，不能自动假设已授权或直接拒绝。",
                reviewer_checkpoints=checkpoints,
            )

        return ShopIdentityResult(
            case_id=item.case_id,
            controlled_brand=brand.brand_name if brand else item.controlled_brand,
            shop_name_signal=name_signal,
            avatar_signal=avatar_signal,
            authorization=item.authorization_status,
            suggested_decision=ProductReviewDecision.APPROVE,
            confidence=ReviewConfidence.MEDIUM,
            reason="未发现足以构成店铺身份冒充的品牌信号。",
        )

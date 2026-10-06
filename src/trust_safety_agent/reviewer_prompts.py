"""Versioned reviewer-owned prompts kept separate from official decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.case_store import JsonlStore
from trust_safety_agent.schema import StrictModel, StringEnum


class PromptOwnerType(StringEnum):
    REVIEWER = "reviewer"
    CANDIDATE = "candidate"


class ReviewerPromptStatus(StringEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ReviewerPromptWorkflow(StringEnum):
    COMPREHENSIVE = "comprehensive"
    SHOP = "shop_identity"
    PRODUCT = "product_ipr"


class ReviewerPromptVersion(StrictModel):
    prompt_id: str = Field(default_factory=lambda: f"RPR-{uuid4().hex[:10].upper()}")
    owner_type: PromptOwnerType = PromptOwnerType.REVIEWER
    version: int = Field(default=1, ge=1)
    name: str = Field(min_length=2, max_length=100)
    workflow: ReviewerPromptWorkflow = ReviewerPromptWorkflow.PRODUCT
    instructions: str = Field(min_length=20, max_length=12000)
    use_policy_rag: bool = True
    use_brand_rag: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_version: Optional[int] = Field(default=None, ge=1)
    status: ReviewerPromptStatus = ReviewerPromptStatus.DRAFT


class ExperimentalPromptResult(StrictModel):
    case_id: str
    prompt_ref: str
    decision: str
    confidence: float = Field(ge=0, le=1)
    reason: str
    recommended_action: str
    trace_id: str


class ReviewerPromptStore:
    def __init__(self, path: Path) -> None:
        self.store = JsonlStore(path, ReviewerPromptVersion)

    def list(self) -> List[ReviewerPromptVersion]:
        return self.store.list()

    def save(self, prompt: ReviewerPromptVersion) -> ReviewerPromptVersion:
        if any(item.prompt_id == prompt.prompt_id and item.version == prompt.version for item in self.list()):
            raise ValueError(f"{prompt.prompt_id} version {prompt.version} already exists")
        self.store.append(prompt)
        return prompt

    def ensure_defaults(self) -> None:
        defaults = (
            ReviewerPromptVersion(
                prompt_id="RPR-DEMO-PRODUCT",
                name="商品证据边界实验",
                workflow=ReviewerPromptWorkflow.PRODUCT,
                instructions="【角色】商品审核辅助员。【目标】检查 Counterfeit、Knockoff、MBA 与 TMI。【步骤】分别检查文本、图片、授权和价格证据，再按政策逐项判断。【边界】单一品牌词或低价不能直接拒绝，关键证据缺失时转人工。【Few-shot】兼容 iPhone 的第三方手机壳且无官方暗示时适用兼容性豁免。【输出】给出结构化标签、证据、置信度和人工检查点。",
                status=ReviewerPromptStatus.ACTIVE,
            ),
            ReviewerPromptVersion(
                prompt_id="RPR-DEMO-SHOP",
                name="店铺身份歧义实验",
                workflow=ReviewerPromptWorkflow.SHOP,
                instructions="【角色】商家身份审核辅助员。【目标】分别判断 Shop Name 与 Shop Avatar 是否存在品牌冒用。【步骤】识别品牌、检查官方身份词、核对授权、处理 Meaningful Word。【边界】普通词、人名和地名不能仅凭词面直接违规，头像不可读或语义有歧义时转人工。【Few-shot】“Coach 教练培训工作室”且无品牌 Logo 时不直接认定品牌冒用。【输出】分别返回名称与头像状态、证据、置信度和补充材料提示。",
                status=ReviewerPromptStatus.ACTIVE,
            ),
            ReviewerPromptVersion(
                prompt_id="RPR-DEMO-PRODUCT",
                version=2,
                parent_version=1,
                name="商品证据边界实验",
                workflow=ReviewerPromptWorkflow.PRODUCT,
                instructions="【角色】商品审核辅助员。【目标】检查 Counterfeit、Knockoff、MBA 与 TMI。【步骤】分别检查文本、图片、授权和价格证据，再按政策逐项判断。【边界】单一品牌词或低价不能直接拒绝，关键证据缺失时转人工。【Few-shot】兼容 iPhone 的第三方手机壳且无官方暗示时适用兼容性豁免。【输出】给出结构化标签、证据、置信度和人工检查点。",
                status=ReviewerPromptStatus.ACTIVE,
            ),
            ReviewerPromptVersion(
                prompt_id="RPR-DEMO-SHOP",
                version=2,
                parent_version=1,
                name="店铺身份歧义实验",
                workflow=ReviewerPromptWorkflow.SHOP,
                instructions="【角色】商家身份审核辅助员。【目标】分别判断 Shop Name 与 Shop Avatar 是否存在品牌冒用。【步骤】识别品牌、检查官方身份词、核对授权、处理 Meaningful Word。【边界】普通词、人名和地名不能仅凭词面直接违规，头像不可读或语义有歧义时转人工。【Few-shot】“Coach 教练培训工作室”且无品牌 Logo 时不直接认定品牌冒用。【输出】分别返回名称与头像状态、证据、置信度和补充材料提示。",
                status=ReviewerPromptStatus.ACTIVE,
            ),
        )
        existing = {(item.prompt_id, item.version) for item in self.list()}
        for prompt in defaults:
            if (prompt.prompt_id, prompt.version) not in existing:
                self.store.append(prompt)

    def new_version(self, prompt_id: str, instructions: str, *, use_policy_rag: Optional[bool] = None, use_brand_rag: Optional[bool] = None) -> ReviewerPromptVersion:
        versions = [item for item in self.list() if item.prompt_id == prompt_id]
        if not versions:
            raise KeyError(prompt_id)
        current = max(versions, key=lambda item: item.version)
        return self.save(ReviewerPromptVersion(
            prompt_id=current.prompt_id,
            owner_type=current.owner_type,
            version=current.version + 1,
            name=current.name,
            workflow=current.workflow,
            instructions=instructions,
            use_policy_rag=current.use_policy_rag if use_policy_rag is None else use_policy_rag,
            use_brand_rag=current.use_brand_rag if use_brand_rag is None else use_brand_rag,
            parent_version=current.version,
            status=ReviewerPromptStatus.DRAFT,
        ))

    def copy_to_candidate(self, prompt_id: str, version: int) -> ReviewerPromptVersion:
        source = next((item for item in self.list() if item.prompt_id == prompt_id and item.version == version), None)
        if source is None:
            raise KeyError(f"{prompt_id}@{version}")
        candidate = source.model_copy(update={"prompt_id": f"CPR-{uuid4().hex[:10].upper()}", "owner_type": PromptOwnerType.CANDIDATE, "version": 1, "parent_version": None, "created_at": datetime.now(timezone.utc), "status": ReviewerPromptStatus.DRAFT})
        return self.save(ReviewerPromptVersion.model_validate(candidate.model_dump()))

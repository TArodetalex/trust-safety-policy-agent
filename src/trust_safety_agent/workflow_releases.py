"""Append-only Workflow release mapping for official Prompt and Skill versions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.case_store import JsonlStore
from trust_safety_agent.prompt_skills import PromptSkillRegistry
from trust_safety_agent.schema import StrictModel, StringEnum


class WorkflowName(StringEnum):
    ROUTER = "comprehensive_router"
    SHOP = "shop_identity"
    PRODUCT = "product_ipr"


class WorkflowRelease(StrictModel):
    release_id: str = Field(default_factory=lambda: f"WFR-{uuid4().hex[:10].upper()}")
    workflow: WorkflowName
    configuration_version: int = Field(ge=1)
    skill_id: str
    skill_version: str
    prompt_id: str
    prompt_version: str
    status: str = Field(default="published", pattern=r"^published$")
    previous_release_id: Optional[str] = None
    note: str = Field(default="", max_length=500)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def skill_ref(self) -> str:
        return f"{self.skill_id}@{self.skill_version}"

    @property
    def prompt_ref(self) -> str:
        return f"{self.prompt_id}@{self.prompt_version}"


DEFAULT_RELEASES: Dict[WorkflowName, tuple[str, str]] = {
    WorkflowName.ROUTER: ("SKL-GENERAL-POLICY", "v2.0.0"),
    WorkflowName.SHOP: ("SKL-GENERAL-POLICY", "v2.0.0"),
    WorkflowName.PRODUCT: ("SKL-PRODUCT-IPR", "v2.0.0"),
}


class WorkflowReleaseStore:
    """Publishing is explicit; saving a Skill never changes a live Workflow."""

    def __init__(self, path: Path) -> None:
        self.store = JsonlStore(path, WorkflowRelease)

    def list(self, workflow: Optional[WorkflowName] = None) -> List[WorkflowRelease]:
        records = self.store.list()
        if workflow is not None:
            records = [item for item in records if item.workflow == workflow]
        return records

    def current(self, workflow: WorkflowName) -> Optional[WorkflowRelease]:
        records = self.list(workflow)
        return max(records, key=lambda item: item.configuration_version) if records else None

    def ensure_defaults(self, registry: PromptSkillRegistry) -> None:
        for workflow, (skill_id, skill_version) in DEFAULT_RELEASES.items():
            if self.current(workflow) is None:
                self.publish(workflow, skill_id, skill_version, registry, note="Phase 9 默认发布配置")

    def publish(
        self,
        workflow: WorkflowName,
        skill_id: str,
        skill_version: str,
        registry: PromptSkillRegistry,
        *,
        note: str = "",
    ) -> WorkflowRelease:
        resolved = registry.resolve_skill(skill_id, skill_version)
        current = self.current(workflow)
        release = WorkflowRelease(
            workflow=workflow,
            configuration_version=(current.configuration_version + 1 if current else 1),
            skill_id=resolved.skill.skill_id,
            skill_version=resolved.skill.version,
            prompt_id=resolved.prompt.prompt_id,
            prompt_version=resolved.prompt.version,
            previous_release_id=current.release_id if current else None,
            note=note.strip(),
        )
        self.store.append(release)
        return release

    def rollback(self, workflow: WorkflowName, registry: PromptSkillRegistry, *, note: str = "") -> WorkflowRelease:
        history = sorted(self.list(workflow), key=lambda item: item.configuration_version)
        if len(history) < 2:
            raise ValueError("当前 Workflow 还没有可回退的历史发布版本")
        target = history[-2]
        return self.publish(
            workflow,
            target.skill_id,
            target.skill_version,
            registry,
            note=note or f"回退到配置 v{target.configuration_version}",
        )

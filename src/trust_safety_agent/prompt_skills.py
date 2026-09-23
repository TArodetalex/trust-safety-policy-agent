"""Versioned Prompt and Skill registry for model-backed workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import Field

from trust_safety_agent.schema import StrictModel, StringEnum


class PromptWorkflow(StringEnum):
    GENERAL_CASE = "general_case"
    CONTROLLED_AGENT = "controlled_agent"


class PromptVersion(StrictModel):
    prompt_id: str = Field(pattern=r"^PRM-[A-Z0-9-]{3,80}$")
    version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    name: str = Field(min_length=2, max_length=100)
    workflow: PromptWorkflow
    instructions: str = Field(min_length=30, max_length=12000)
    change_note: str = Field(min_length=3, max_length=500)


class SkillDefinition(StrictModel):
    skill_id: str = Field(pattern=r"^SKL-[A-Z0-9-]{3,80}$")
    version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    name: str = Field(min_length=2, max_length=100)
    description: str = Field(min_length=10, max_length=500)
    workflow: PromptWorkflow
    prompt_id: str
    prompt_version: str
    allowed_tools: List[str] = Field(default_factory=list, max_length=12)
    policy_scope: List[str] = Field(default_factory=list, max_length=20)
    max_steps: int = Field(default=4, ge=1, le=8)
    output_contract: str = Field(min_length=2, max_length=100)


class ResolvedSkill(StrictModel):
    skill: SkillDefinition
    prompt: PromptVersion


class PromptSkillRegistry:
    def __init__(
        self,
        prompt_presets_path: Path,
        skill_presets_path: Path,
        custom_prompts_path: Path,
        custom_skills_path: Path,
    ) -> None:
        self.prompt_presets_path = prompt_presets_path
        self.skill_presets_path = skill_presets_path
        self.custom_prompts_path = custom_prompts_path
        self.custom_skills_path = custom_skills_path

    @staticmethod
    def _preset_records(path: Path, key: str) -> list[dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get(key)
        if not isinstance(records, list):
            raise ValueError(f"{path.name} must contain a {key} list")
        return records

    @staticmethod
    def _custom_records(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def prompts(self, workflow: Optional[PromptWorkflow] = None) -> List[PromptVersion]:
        records = [
            PromptVersion.model_validate(item)
            for item in (
                self._preset_records(self.prompt_presets_path, "prompts")
                + self._custom_records(self.custom_prompts_path)
            )
        ]
        if workflow is not None:
            records = [item for item in records if item.workflow == workflow]
        return sorted(records, key=lambda item: (item.prompt_id, item.version))

    def skills(self, workflow: Optional[PromptWorkflow] = None) -> List[SkillDefinition]:
        records = [
            SkillDefinition.model_validate(item)
            for item in (
                self._preset_records(self.skill_presets_path, "skills")
                + self._custom_records(self.custom_skills_path)
            )
        ]
        if workflow is not None:
            records = [item for item in records if item.workflow == workflow]
        return sorted(records, key=lambda item: (item.skill_id, item.version))

    def get_prompt(self, prompt_id: str, version: str) -> PromptVersion:
        match = next(
            (
                item
                for item in self.prompts()
                if item.prompt_id == prompt_id and item.version == version
            ),
            None,
        )
        if match is None:
            raise KeyError(f"prompt {prompt_id}@{version} was not found")
        return match

    def resolve_skill(self, skill_id: str, version: str) -> ResolvedSkill:
        skill = next(
            (
                item
                for item in self.skills()
                if item.skill_id == skill_id and item.version == version
            ),
            None,
        )
        if skill is None:
            raise KeyError(f"skill {skill_id}@{version} was not found")
        return ResolvedSkill(
            skill=skill,
            prompt=self.get_prompt(skill.prompt_id, skill.prompt_version),
        )

    @staticmethod
    def _append_unique(path: Path, key: tuple[str, str], records: list, model) -> None:
        if any((getattr(item, key[0]), item.version) == (key[1], model.version) for item in records):
            raise ValueError(f"{key[1]}@{model.version} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(model.model_dump_json() + "\n")

    def save_prompt(self, prompt: PromptVersion) -> None:
        self._append_unique(
            self.custom_prompts_path,
            ("prompt_id", prompt.prompt_id),
            self.prompts(),
            prompt,
        )

    def save_skill(self, skill: SkillDefinition) -> None:
        prompt = self.get_prompt(skill.prompt_id, skill.prompt_version)
        if prompt.workflow != skill.workflow:
            raise ValueError("skill workflow must match its prompt workflow")
        self._append_unique(
            self.custom_skills_path,
            ("skill_id", skill.skill_id),
            self.skills(),
            skill,
        )

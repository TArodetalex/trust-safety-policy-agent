from pathlib import Path

import pytest

from trust_safety_agent.config import (
    DEFAULT_PROMPT_PRESETS_PATH,
    DEFAULT_SKILL_PRESETS_PATH,
)
from trust_safety_agent.prompt_skills import (
    PromptSkillRegistry,
    PromptVersion,
    PromptWorkflow,
    SkillDefinition,
)


def registry(tmp_path: Path) -> PromptSkillRegistry:
    return PromptSkillRegistry(
        DEFAULT_PROMPT_PRESETS_PATH,
        DEFAULT_SKILL_PRESETS_PATH,
        tmp_path / "custom_prompts.jsonl",
        tmp_path / "custom_skills.jsonl",
    )


def test_resolves_knockoff_skill_to_versioned_prompt(tmp_path: Path) -> None:
    resolved = registry(tmp_path).resolve_skill("SKL-KNOCKOFF-REVIEW", "v1.0.0")

    assert resolved.prompt.prompt_id == "PRM-KNOCKOFF-FOCUS"
    assert "POL-KO-001" in resolved.prompt.instructions
    assert "classify_policy_signals" in resolved.skill.allowed_tools
    assert resolved.skill.policy_scope == ["POL-KO-001", "POL-EX-001"]


def test_saves_custom_prompt_and_skill_versions(tmp_path: Path) -> None:
    store = registry(tmp_path)
    prompt = PromptVersion(
        prompt_id="PRM-TEST-CUSTOM",
        version="v1.0.0",
        name="测试 Prompt",
        workflow=PromptWorkflow.CONTROLLED_AGENT,
        instructions="Require policy evidence and route uncertain cases to manual review.",
        change_note="Initial test version.",
    )
    store.save_prompt(prompt)
    store.save_skill(
        SkillDefinition(
            skill_id="SKL-TEST-CUSTOM",
            version="v1.0.0",
            name="测试 Skill",
            description="A versioned skill created for registry tests.",
            workflow=PromptWorkflow.CONTROLLED_AGENT,
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.version,
            allowed_tools=["classify_policy_signals"],
            policy_scope=["POL-KO-001"],
            max_steps=2,
            output_contract="AgentRunResult",
        )
    )

    resolved = store.resolve_skill("SKL-TEST-CUSTOM", "v1.0.0")
    assert resolved.prompt == prompt
    with pytest.raises(ValueError, match="already exists"):
        store.save_prompt(prompt)


def test_skill_cannot_bind_prompt_from_another_workflow(tmp_path: Path) -> None:
    store = registry(tmp_path)
    with pytest.raises(ValueError, match="workflow must match"):
        store.save_skill(
            SkillDefinition(
                skill_id="SKL-WRONG-WORKFLOW",
                version="v1.0.0",
                name="错误工作流",
                description="This skill intentionally binds the wrong workflow.",
                workflow=PromptWorkflow.GENERAL_CASE,
                prompt_id="PRM-KNOCKOFF-FOCUS",
                prompt_version="v1.0.0",
                allowed_tools=[],
                policy_scope=[],
                max_steps=1,
                output_contract="LLMDecisionDraft",
            )
        )

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


def test_phase8_skill_consumes_structured_product_evidence(tmp_path: Path) -> None:
    resolved = registry(tmp_path).resolve_skill("SKL-PRODUCT-IPR", "v2.0.0")

    assert resolved.prompt.version == "v2.0.0"
    assert "review_product" in resolved.skill.allowed_tools
    assert {"POL-MBA-001", "POL-TMI-001", "POL-ER-001"}.issubset(
        resolved.skill.policy_scope
    )
    assert "logo_match_type" in resolved.prompt.instructions
    assert "brand_authorization_status" in resolved.prompt.instructions


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


def test_revises_prompt_and_skill_without_overwriting_history(tmp_path: Path) -> None:
    store = registry(tmp_path)
    source_prompt = store.get_prompt("PRM-PRODUCT-SAFE", "v2.0.0")
    revised_prompt = store.revise_prompt(
        source_prompt,
        name="商品知识产权综合审核（候选）",
        instructions=source_prompt.instructions + "\n补充：图片不清晰时必须转人工。",
        change_note="补充图片质量边界",
    )
    assert revised_prompt.version == "v2.0.1"
    assert store.get_prompt("PRM-PRODUCT-SAFE", "v2.0.0") == source_prompt

    source_skill = next(item for item in store.skills() if item.skill_id == "SKL-PRODUCT-IPR" and item.version == "v2.0.0")
    revised_skill = store.revise_skill(
        source_skill,
        name="商品知识产权综合审核 Skill（候选）",
        description=source_skill.description + " 增加图片质量检查。",
        prompt_id=revised_prompt.prompt_id,
        prompt_version=revised_prompt.version,
        allowed_tools=source_skill.allowed_tools,
        policy_scope=source_skill.policy_scope,
        max_steps=source_skill.max_steps,
        output_contract=source_skill.output_contract,
    )
    assert revised_skill.version == "v2.0.1"
    assert store.resolve_skill(revised_skill.skill_id, revised_skill.version).prompt == revised_prompt

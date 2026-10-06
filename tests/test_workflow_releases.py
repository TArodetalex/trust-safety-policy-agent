from pathlib import Path

from trust_safety_agent.config import DEFAULT_PROMPT_PRESETS_PATH, DEFAULT_SKILL_PRESETS_PATH
from trust_safety_agent.prompt_skills import PromptSkillRegistry
from trust_safety_agent.workflow_releases import WorkflowName, WorkflowReleaseStore


def registry(tmp_path: Path) -> PromptSkillRegistry:
    return PromptSkillRegistry(
        DEFAULT_PROMPT_PRESETS_PATH,
        DEFAULT_SKILL_PRESETS_PATH,
        tmp_path / "prompts.jsonl",
        tmp_path / "skills.jsonl",
    )


def test_skill_creation_does_not_publish_and_rollback_is_append_only(tmp_path: Path) -> None:
    store = WorkflowReleaseStore(tmp_path / "releases.jsonl")
    prompt_registry = registry(tmp_path)
    store.ensure_defaults(prompt_registry)
    original = store.current(WorkflowName.PRODUCT)
    assert original is not None

    changed = store.publish(
        WorkflowName.PRODUCT,
        "SKL-KNOCKOFF-REVIEW",
        "v2.0.0",
        prompt_registry,
        note="candidate passed evaluation",
    )
    assert changed.configuration_version == original.configuration_version + 1
    assert changed.skill_id == "SKL-KNOCKOFF-REVIEW"

    rolled_back = store.rollback(WorkflowName.PRODUCT, prompt_registry)
    assert rolled_back.configuration_version == changed.configuration_version + 1
    assert rolled_back.skill_ref == original.skill_ref
    assert len(store.list(WorkflowName.PRODUCT)) == 3

"""One-case live smoke test for the bounded tool-calling agent."""

from __future__ import annotations

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import (
    DEFAULT_BRAND_LIBRARY_PATH,
    DEFAULT_CHROMA_DIRECTORY,
    DEFAULT_CUSTOM_PROMPTS_PATH,
    DEFAULT_CUSTOM_SKILLS_PATH,
    DEFAULT_POLICY_PATH,
    DEFAULT_PROMPT_PRESETS_PATH,
    DEFAULT_REVIEW_RECORDS_PATH,
    DEFAULT_SKILL_PRESETS_PATH,
)
from trust_safety_agent.controlled_agent import AgentRequest, ControlledPolicyAgent
from trust_safety_agent.controlled_tools import build_default_tool_registry
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.prompt_skills import PromptSkillRegistry
from trust_safety_agent.resilient_llm_client import (
    ResilientLLMSettings,
    ResilientOpenAICompatibleClient,
)
from trust_safety_agent.reviewer_workspace import ReviewerWorkspaceStore
from trust_safety_agent.vector_store import PolicyVectorStore


def main() -> None:
    settings = ResilientLLMSettings.from_env()
    if not settings.configured:
        raise SystemExit("LLM settings are not configured")
    store = PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)
    if store.count() == 0:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))
    registry = build_default_tool_registry(
        store,
        ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH),
        ReviewerWorkspaceStore(DEFAULT_REVIEW_RECORDS_PATH),
    )
    skill_registry = PromptSkillRegistry(
        DEFAULT_PROMPT_PRESETS_PATH,
        DEFAULT_SKILL_PRESETS_PATH,
        DEFAULT_CUSTOM_PROMPTS_PATH,
        DEFAULT_CUSTOM_SKILLS_PATH,
    )
    resolved = skill_registry.resolve_skill("SKL-KNOCKOFF-REVIEW", "v1.0.0")
    case_data = {
        "case_id": "AGENT-SMOKE-001",
        "title": "Gucci handbag",
        "description": "Gucci dupe, same design as the original, not authentic",
        "optional_brand_field": "Gucci",
        "ocr_text": "Gucci",
        "visual_marks": ["Gucci"],
    }
    result = ControlledPolicyAgent(
        ResilientOpenAICompatibleClient(settings),
        registry,
        max_steps=resolved.skill.max_steps,
        allowed_tools=resolved.skill.allowed_tools,
        additional_instructions=resolved.prompt.instructions,
        policy_scope=resolved.skill.policy_scope,
    ).run(
        AgentRequest(
            case_id=case_data["case_id"],
            task="Use the available tools to review this product safely.",
            case_data=case_data,
        )
    )
    print(
        f"agent smoke completed: decision={result.decision.value}; "
        f"stop={result.stop_reason.value}; tools={len(result.tool_executions)}; "
        f"tokens={result.total_tokens}; "
        f"skill={resolved.skill.skill_id}@{resolved.skill.version}"
    )
    print(f"reason={result.reason[:500]}")
    for execution in result.tool_executions:
        print(f"step={execution.step}; tool={execution.tool_name}; status={execution.status.value}")


if __name__ == "__main__":
    main()

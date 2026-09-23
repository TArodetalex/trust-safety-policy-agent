from pathlib import Path

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_POLICY_PATH
from trust_safety_agent.controlled_agent import (
    AgentRequest,
    AgentStopReason,
    ControlledPolicyAgent,
)
from trust_safety_agent.controlled_tools import (
    ToolExecutionStatus,
    build_default_tool_registry,
)
from trust_safety_agent.llm_core import LLMCallMetadata
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.product_review import ProductReviewDecision
from trust_safety_agent.reviewer_workspace import ReviewerWorkspaceStore
from trust_safety_agent.vector_store import PolicyVectorStore


class FakePlanner:
    model = "fake-planner"

    def __init__(self, plans):
        self.plans = list(plans)
        self.last_metadata = LLMCallMetadata(
            requested_model=self.model,
            total_tokens=10,
        )

    def complete_json(self, **_kwargs):
        return self.plans.pop(0)


def registry(tmp_path: Path):
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    return build_default_tool_registry(
        store,
        ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH),
        ReviewerWorkspaceStore(tmp_path / "reviews.jsonl"),
    )


def product_arguments():
    return {
        "case_id": "AGENT-001",
        "title": "Gucci handbag",
        "description": "1:1 mirror copy",
        "optional_brand_field": "Gucci",
        "ocr_text": "",
        "visual_marks": ["Gucci"],
    }


def test_controlled_agent_executes_tools_and_allows_grounded_reject(tmp_path: Path):
    planner = FakePlanner(
        [
            {
                "action": "review_product",
                "arguments": product_arguments(),
            },
            {
                "action": "tool_call",
                "rationale": "Retrieve policy evidence.",
                "tool_name": "search_policy",
                "tool_arguments": {"query": "Gucci replica counterfeit", "top_k": 5},
            },
            {
                "action": "final",
                "rationale": "Evidence supports rejection.",
                "decision": "reject",
                "confidence": 0.91,
                "policy_id": "POL-CF-001",
                "reason": "Controlled-brand counterfeit language is policy-grounded.",
            },
        ]
    )
    result = ControlledPolicyAgent(planner, registry(tmp_path)).run(
        AgentRequest(
            case_id="AGENT-001",
            task="Review this product safely.",
            case_data=product_arguments(),
        )
    )
    assert result.decision == ProductReviewDecision.REJECT
    assert result.stop_reason == AgentStopReason.COMPLETED
    assert len(result.tool_executions) == 2
    assert result.total_tokens == 30


def test_controlled_agent_blocks_write_tool(tmp_path: Path):
    planner = FakePlanner(
        [
            {
                "action": "tool_call",
                "rationale": "Try to create a review.",
                "tool_name": "submit_human_review",
                "tool_arguments": {
                    "case_id": "AGENT-002",
                    "case_input": {},
                    "agent_decision": "manual_review",
                    "agent_confidence": "low",
                    "agent_reason": "Need review",
                    "user_confirmed": True,
                },
            }
        ]
    )
    agent = ControlledPolicyAgent(
        planner,
        registry(tmp_path),
        allowed_tools=["review_product", "submit_human_review"],
    )
    result = agent.run(
        AgentRequest(case_id="AGENT-002", task="Review safely.", case_data={})
    )
    assert result.decision == ProductReviewDecision.MANUAL_REVIEW
    assert result.stop_reason == AgentStopReason.TOOL_FAILURE
    assert result.tool_executions[0].status == ToolExecutionStatus.DENIED


def test_controlled_agent_blocks_ungrounded_final_decision(tmp_path: Path):
    planner = FakePlanner(
        [
            {
                "action": "tool_call",
                "rationale": "Look up a brand.",
                "tool_name": "lookup_brand",
                "tool_arguments": {"text": "Gucci"},
            },
            {
                "action": "final",
                "rationale": "Reject.",
                "decision": "reject",
                "confidence": 0.99,
                "policy_id": "POL-CF-001",
                "reason": "Brand found.",
            },
        ]
    )
    result = ControlledPolicyAgent(planner, registry(tmp_path)).run(
        AgentRequest(case_id="AGENT-003", task="Review safely.", case_data={})
    )
    assert result.decision == ProductReviewDecision.MANUAL_REVIEW
    assert result.stop_reason == AgentStopReason.GUARDRAIL


def test_tool_registry_requires_both_write_approval_and_confirmation(tmp_path: Path):
    tools = registry(tmp_path)
    arguments = {
        "case_id": "AGENT-004",
        "case_input": {"title": "unknown"},
        "agent_decision": "manual_review",
        "agent_confidence": "low",
        "agent_reason": "Evidence missing.",
        "user_confirmed": False,
    }
    denied = tools.execute(
        step=1,
        tool_name="submit_human_review",
        arguments=arguments,
        allowed_tools=tools.names(),
        allow_write=False,
    )
    invalid_confirmation = tools.execute(
        step=1,
        tool_name="submit_human_review",
        arguments=arguments,
        allowed_tools=tools.names(),
        allow_write=True,
    )
    assert denied.status == ToolExecutionStatus.DENIED
    assert invalid_confirmation.status == ToolExecutionStatus.ERROR


def test_policy_signal_tool_returns_grounded_knockoff_evidence(tmp_path: Path):
    tools = registry(tmp_path)
    result = tools.execute(
        step=1,
        tool_name="classify_policy_signals",
        arguments={
            "content_type": "product",
            "input_text": "Gucci dupe, same design as the original",
        },
        allowed_tools=tools.names(),
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.output["decision"] == "reject"
    assert result.output["policy_label"] == "knockoff"
    assert result.output["matched_policy"][0]["policy_id"] == "POL-KO-001"


def test_skill_policy_scope_blocks_out_of_scope_reject(tmp_path: Path):
    planner = FakePlanner(
        [
            {
                "action": "search_policy",
                "arguments": {"query": "counterfeit replica", "top_k": 5},
            },
            {
                "action": "final",
                "rationale": "Counterfeit evidence found.",
                "decision": "reject",
                "confidence": 0.95,
                "policy_id": "POL-CF-001",
                "reason": "Counterfeit policy matched.",
            },
        ]
    )
    result = ControlledPolicyAgent(
        planner,
        registry(tmp_path),
        allowed_tools=["search_policy"],
        policy_scope=["POL-KO-001"],
        additional_instructions="Only review knockoff cases.",
    ).run(AgentRequest(case_id="AGENT-005", task="Review safely.", case_data={}))

    assert result.decision == ProductReviewDecision.MANUAL_REVIEW
    assert result.stop_reason == AgentStopReason.GUARDRAIL
    assert "政策范围" in result.reason


def test_agent_derives_confidence_from_policy_signal_tool(tmp_path: Path):
    planner = FakePlanner(
        [
            {
                "action": "classify_policy_signals",
                "arguments": {
                    "content_type": "product",
                    "input_text": "Gucci dupe, same design as the original",
                },
            },
            {
                "action": "reject",
                "policy_id": "POL-KO-001",
                "reason": "The deterministic policy signal is grounded.",
            },
        ]
    )
    result = ControlledPolicyAgent(
        planner,
        registry(tmp_path),
        allowed_tools=["classify_policy_signals"],
        policy_scope=["POL-KO-001"],
    ).run(AgentRequest(case_id="AGENT-006", task="Review knockoff risk.", case_data={}))

    assert result.decision == ProductReviewDecision.REJECT
    assert result.stop_reason == AgentStopReason.COMPLETED
    assert result.confidence.value == "high"

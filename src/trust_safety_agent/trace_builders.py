"""Trace adapters for deterministic Product and Shop review workflows."""

from __future__ import annotations

from trust_safety_agent.product_review import ProductReviewInput, ProductReviewResult
from trust_safety_agent.controlled_agent import AgentRequest, AgentRunResult
from trust_safety_agent.controlled_tools import ToolExecutionStatus
from trust_safety_agent.shop_identity import ShopIdentityInput, ShopIdentityResult
from trust_safety_agent.trace import (
    ExecutionTrace,
    TraceNode,
    TraceNodeStatus,
    build_trace,
)


def product_review_trace(
    item: ProductReviewInput,
    result: ProductReviewResult,
    latency_ms: int,
) -> ExecutionTrace:
    nodes = [
        TraceNode(
            node_name="Input",
            input_summary=item.model_dump(mode="json"),
            output={"normalized": True},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Candidate Recall",
            input_summary={"sources": ["brand_field", "title", "description", "ocr", "image_visual"]},
            output={"candidates": [item.model_dump(mode="json") for item in result.candidate_brands]},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Retrieval",
            input_summary={"controlled_brands": result.controlled_brand_matches},
            output={"policy_references": [item.model_dump(mode="json") for item in result.policy_references]},
            status=(TraceNodeStatus.SUCCESS if result.policy_references else TraceNodeStatus.SKIPPED),
            latency_ms=0,
        ),
        TraceNode(
            node_name="Evidence",
            output={"evidence": [item.model_dump(mode="json") for item in result.evidence]},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Exemption",
            output={"possible_exemptions": [item.value for item in result.possible_exemptions]},
            status=(TraceNodeStatus.SUCCESS if result.possible_exemptions else TraceNodeStatus.SKIPPED),
            latency_ms=0,
        ),
        TraceNode(
            node_name="Judge",
            output={"decision": result.suggested_decision.value, "reason": result.reason},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=latency_ms,
            confidence={"high": 0.9, "medium": 0.6, "low": 0.3}[result.confidence.value],
        ),
        TraceNode(
            node_name="Guardrail",
            output={"reviewer_checkpoints": result.reviewer_checkpoints},
            status=(TraceNodeStatus.FALLBACK if result.suggested_decision.value == "manual_review" else TraceNodeStatus.SUCCESS),
            latency_ms=0,
        ),
        TraceNode(
            node_name="Final",
            output={"suggested_decision": result.suggested_decision.value},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
    ]
    return build_trace(
        case_id=item.case_id,
        workflow="product_ipr",
        nodes=nodes,
        final_output=result.model_dump(mode="json"),
    )


def shop_identity_trace(
    item: ShopIdentityInput,
    result: ShopIdentityResult,
    latency_ms: int,
) -> ExecutionTrace:
    nodes = [
        TraceNode(
            node_name="Input",
            input_summary=item.model_dump(mode="json"),
            output={"normalized": True},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Authorization Gate",
            input_summary={"authorization_status": item.authorization_status.value},
            output={"exemptions": result.possible_exemptions},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Shop Name Judge",
            input_summary={"shop_name": item.shop_name},
            output=result.shop_name_signal.model_dump(mode="json"),
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        ),
        TraceNode(
            node_name="Avatar Judge",
            input_summary={"avatar_url": str(item.avatar_url or "")},
            output=result.avatar_signal.model_dump(mode="json"),
            status=(TraceNodeStatus.FALLBACK if result.avatar_signal.signal_type.value == "unavailable_image" else TraceNodeStatus.SUCCESS),
            latency_ms=0,
        ),
        TraceNode(
            node_name="Retrieval",
            output={"policy_references": [item.model_dump(mode="json") for item in result.policy_references]},
            status=(TraceNodeStatus.SUCCESS if result.policy_references else TraceNodeStatus.SKIPPED),
            latency_ms=0,
        ),
        TraceNode(
            node_name="Final",
            output={"decision": result.suggested_decision.value, "reason": result.reason},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=latency_ms,
            confidence={"high": 0.9, "medium": 0.6, "low": 0.3}[result.confidence.value],
        ),
    ]
    return build_trace(
        case_id=item.case_id,
        workflow="shop_identity",
        nodes=nodes,
        final_output=result.model_dump(mode="json"),
    )


def controlled_agent_trace(
    request: AgentRequest,
    result: AgentRunResult,
) -> ExecutionTrace:
    status_map = {
        ToolExecutionStatus.SUCCESS: TraceNodeStatus.SUCCESS,
        ToolExecutionStatus.DENIED: TraceNodeStatus.FALLBACK,
        ToolExecutionStatus.INVALID: TraceNodeStatus.ERROR,
        ToolExecutionStatus.ERROR: TraceNodeStatus.ERROR,
    }
    nodes = [
        TraceNode(
            node_name="Agent Input",
            input_summary=request.model_dump(mode="json"),
            output={"allowed_workflow": "product_review"},
            status=TraceNodeStatus.SUCCESS,
            latency_ms=0,
        )
    ]
    nodes.extend(
        TraceNode(
            node_name=f"Tool · {execution.tool_name}",
            input_summary=execution.arguments,
            output=execution.output,
            status=status_map[execution.status],
            latency_ms=execution.latency_ms,
            error=(
                execution.error
                if execution.status in {ToolExecutionStatus.INVALID, ToolExecutionStatus.ERROR}
                else None
            ),
        )
        for execution in result.tool_executions
    )
    nodes.append(
        TraceNode(
            node_name="Agent Guardrail & Final",
            input_summary={"stop_reason": result.stop_reason.value},
            output={
                "decision": result.decision.value,
                "policy_id": result.policy_id,
                "reason": result.reason,
                "total_tokens": result.total_tokens,
            },
            status=(
                TraceNodeStatus.SUCCESS
                if result.stop_reason.value == "completed"
                else TraceNodeStatus.FALLBACK
            ),
            latency_ms=0,
            confidence={"high": 0.9, "medium": 0.6, "low": 0.3}[result.confidence.value],
        )
    )
    return build_trace(
        case_id=request.case_id,
        workflow="controlled_tool_calling_agent",
        nodes=nodes,
        final_output=result.model_dump(mode="json"),
    )

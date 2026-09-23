"""Typed, permission-aware tools shared by the agent and MCP server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import Field

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.product_review import (
    ProductReviewAssistant,
    ProductReviewDecision,
    ProductReviewInput,
    ProductReviewResult,
    ReviewConfidence,
)
from trust_safety_agent.reviewer_workspace import ReviewWorkflow, ReviewerWorkspaceStore
from trust_safety_agent.schema import ContentType, StrictModel, StringEnum
from trust_safety_agent.trace import sanitize_sensitive
from trust_safety_agent.vector_store import PolicyVectorStore


class ToolPermission(StringEnum):
    READ = "read"
    WRITE = "write"


class ToolExecutionStatus(StringEnum):
    SUCCESS = "success"
    DENIED = "denied"
    INVALID = "invalid"
    ERROR = "error"


class PolicySearchInput(StrictModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=8)


class PolicySearchHit(StrictModel):
    policy_id: str
    chunk_id: str
    title: str
    excerpt: str
    score: float = Field(ge=0, le=1)


class PolicySearchOutput(StrictModel):
    query: str
    hits: List[PolicySearchHit]


class BrandLookupInput(StrictModel):
    text: str = Field(min_length=1, max_length=4000)


class BrandLookupOutput(StrictModel):
    matched_brands: List[str]
    mentions: List[Dict[str, Any]]


class PolicySignalInput(StrictModel):
    content_type: ContentType = ContentType.PRODUCT
    input_text: str = Field(min_length=1, max_length=4000)


class PolicySignalOutput(StrictModel):
    decision: str
    policy_label: Optional[str] = None
    matched_policy: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0, le=1)


class ReviewQueueInput(StrictModel):
    limit: int = Field(default=20, ge=1, le=100)


class ReviewQueueOutput(StrictModel):
    pending_count: int
    records: List[Dict[str, Any]]


class SubmitReviewInput(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    case_input: Dict[str, Any]
    agent_decision: ProductReviewDecision
    agent_confidence: ReviewConfidence
    agent_reason: str = Field(min_length=1, max_length=4000)
    user_confirmed: bool = False


class SubmitReviewOutput(StrictModel):
    review_id: str
    case_id: str
    status: str


class ToolExecution(StrictModel):
    step: int = Field(ge=1)
    tool_name: str
    permission: ToolPermission
    arguments: Dict[str, Any]
    status: ToolExecutionStatus
    output: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)
    error: Optional[str] = Field(default=None, max_length=1000)


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    permission: ToolPermission
    input_model: Type[StrictModel]
    output_model: Type[StrictModel]
    handler: Callable[[StrictModel], StrictModel]

    def public_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission.value,
            "input_schema": self.input_model.model_json_schema(),
        }


class ControlledToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} is already registered")
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        allowed = set(allowed_tools or self._tools)
        return [
            self._tools[name].public_schema()
            for name in sorted(allowed)
            if name in self._tools
        ]

    def execute(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: Dict[str, Any],
        allowed_tools: List[str],
        allow_write: bool = False,
    ) -> ToolExecution:
        started = time.perf_counter()
        tool = self._tools.get(tool_name)
        permission = tool.permission if tool else ToolPermission.READ

        def result(
            status: ToolExecutionStatus,
            *,
            output: Optional[Dict[str, Any]] = None,
            error: Optional[str] = None,
        ) -> ToolExecution:
            return ToolExecution(
                step=step,
                tool_name=tool_name,
                permission=permission,
                arguments=sanitize_sensitive(arguments),
                status=status,
                output=sanitize_sensitive(output or {}),
                latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
                error=error,
            )

        if tool is None or tool_name not in allowed_tools:
            return result(ToolExecutionStatus.DENIED, error="tool is not on the workflow allowlist")
        if tool.permission == ToolPermission.WRITE and not allow_write:
            return result(ToolExecutionStatus.DENIED, error="write tool requires explicit caller approval")
        try:
            validated = tool.input_model.model_validate(arguments)
        except ValueError as exc:
            return result(ToolExecutionStatus.INVALID, error=str(exc)[:1000])
        try:
            output_model = tool.output_model.model_validate(tool.handler(validated))
        except Exception as exc:
            return result(
                ToolExecutionStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
        return result(ToolExecutionStatus.SUCCESS, output=output_model.model_dump(mode="json"))


def build_default_tool_registry(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
) -> ControlledToolRegistry:
    registry = ControlledToolRegistry()
    product_assistant = ProductReviewAssistant(library, store)

    def search_policy(data: PolicySearchInput) -> PolicySearchOutput:
        hits = store.retrieve(data.query, top_k=data.top_k)
        return PolicySearchOutput(
            query=data.query,
            hits=[
                PolicySearchHit(
                    policy_id=hit.chunk.policy_id,
                    chunk_id=hit.chunk.chunk_id,
                    title=hit.chunk.title,
                    excerpt=hit.chunk.content[:500],
                    score=hit.score,
                )
                for hit in hits
            ],
        )

    def lookup_brand(data: BrandLookupInput) -> BrandLookupOutput:
        mentions = library.find_mentions(data.text)
        return BrandLookupOutput(
            matched_brands=sorted({item.brand_name for item in mentions}),
            mentions=[item.model_dump(mode="json") for item in mentions],
        )

    def review_product(data: ProductReviewInput) -> ProductReviewResult:
        return product_assistant.review(data)

    def classify_policy_signals(data: PolicySignalInput) -> PolicySignalOutput:
        result = PolicyAdjudicator(store).adjudicate(
            case_id="C000",
            content_type=data.content_type,
            input_text=data.input_text,
        )
        return PolicySignalOutput(
            decision=result.decision.value,
            policy_label=result.policy_label.value if result.policy_label else None,
            matched_policy=[
                item.model_dump(mode="json") for item in result.matched_policy
            ],
            reason=result.reason,
            confidence=result.confidence,
        )

    def get_review_queue(data: ReviewQueueInput) -> ReviewQueueOutput:
        pending = [item for item in review_store.list_records() if item.reviewer_label is None]
        selected = pending[: data.limit]
        return ReviewQueueOutput(
            pending_count=len(pending),
            records=[
                {
                    "review_id": item.review_id,
                    "case_id": item.case_id,
                    "workflow": item.workflow.value,
                    "agent_decision": item.agent_decision.value,
                    "agent_confidence": item.agent_confidence.value,
                }
                for item in selected
            ],
        )

    def submit_review(data: SubmitReviewInput) -> SubmitReviewOutput:
        if not data.user_confirmed:
            raise ValueError("user_confirmed must be true for this write operation")
        record = review_store.enqueue(
            case_id=data.case_id,
            workflow=ReviewWorkflow.GENERAL_CASE,
            case_input=data.case_input,
            agent_decision=data.agent_decision,
            agent_confidence=data.agent_confidence,
            agent_reason=data.agent_reason,
        )
        return SubmitReviewOutput(
            review_id=record.review_id,
            case_id=record.case_id,
            status="pending",
        )

    for tool in (
        RegisteredTool(
            "search_policy",
            "Search versioned policy chunks and return grounded evidence.",
            ToolPermission.READ,
            PolicySearchInput,
            PolicySearchOutput,
            search_policy,
        ),
        RegisteredTool(
            "lookup_brand",
            "Find boundary-safe controlled-brand mentions in supplied text.",
            ToolPermission.READ,
            BrandLookupInput,
            BrandLookupOutput,
            lookup_brand,
        ),
        RegisteredTool(
            "review_product",
            "Run the deterministic Product IPR review pipeline.",
            ToolPermission.READ,
            ProductReviewInput,
            ProductReviewResult,
            review_product,
        ),
        RegisteredTool(
            "classify_policy_signals",
            "Classify deterministic policy signals, including knockoff, and return evidence.",
            ToolPermission.READ,
            PolicySignalInput,
            PolicySignalOutput,
            classify_policy_signals,
        ),
        RegisteredTool(
            "get_review_queue",
            "List pending human-review tasks without exposing full case payloads.",
            ToolPermission.READ,
            ReviewQueueInput,
            ReviewQueueOutput,
            get_review_queue,
        ),
        RegisteredTool(
            "submit_human_review",
            "Create a review task. Requires caller write approval and user confirmation.",
            ToolPermission.WRITE,
            SubmitReviewInput,
            SubmitReviewOutput,
            submit_review,
        ),
    ):
        registry.register(tool)
    return registry

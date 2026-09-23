"""Structured execution traces with recursive sensitive-field filtering."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field, model_validator

from trust_safety_agent.schema import StrictModel, StringEnum


class TraceNodeStatus(StringEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FALLBACK = "fallback"
    ERROR = "error"


class TraceTokenUsage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "TraceTokenUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class TraceNode(StrictModel):
    node_name: str = Field(min_length=1, max_length=100)
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    status: TraceNodeStatus
    latency_ms: int = Field(ge=0)
    token_usage: Optional[TraceTokenUsage] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    error: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_error(self) -> "TraceNode":
        if self.status == TraceNodeStatus.ERROR and not self.error:
            raise ValueError("error nodes require an error message")
        return self


class ExecutionTrace(StrictModel):
    trace_id: str = Field(pattern=r"^TRC-[A-F0-9]{12}$")
    case_id: str = Field(min_length=1, max_length=100)
    workflow: str = Field(min_length=1, max_length=100)
    nodes: List[TraceNode] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_SENSITIVE_KEYS = re.compile(
    r"^(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|"
    r"secret|client[_-]?secret|credential|authorization[_-]?header|cookie)$",
    re.IGNORECASE,
)
_SENSITIVE_VALUES = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._-]{8,}|\bsk-[A-Za-z0-9._-]{8,})",
    re.IGNORECASE,
)


def sanitize_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE_KEYS.match(str(key))
                else sanitize_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_sensitive(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUES.sub("<redacted>", value)
    return value


def build_trace(
    *,
    case_id: str,
    workflow: str,
    nodes: List[TraceNode],
    final_output: Dict[str, Any],
) -> ExecutionTrace:
    sanitized_nodes = [
        TraceNode.model_validate(sanitize_sensitive(node.model_dump()))
        for node in nodes
    ]
    return ExecutionTrace(
        trace_id=f"TRC-{uuid4().hex[:12].upper()}",
        case_id=case_id,
        workflow=workflow,
        nodes=sanitized_nodes,
        final_output=sanitize_sensitive(final_output),
    )


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: ExecutionTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(trace.model_dump_json() + "\n")

    def list_traces(self, case_id: Optional[str] = None) -> List[ExecutionTrace]:
        if not self.path.exists():
            return []
        traces = [
            ExecutionTrace.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if case_id is not None:
            traces = [trace for trace in traces if trace.case_id == case_id]
        return traces

"""Bad-case queue, confirmed attribution, and immutable replay comparison."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.case_store import JsonlStore
from trust_safety_agent.schema import StrictModel, StringEnum
from trust_safety_agent.trace import ExecutionTrace


class FailureType(StringEnum):
    INPUT_MISSING = "input_missing"
    NORMALIZATION_ERROR = "normalization_error"
    EVIDENCE_EXTRACTION_ERROR = "evidence_extraction_error"
    CANDIDATE_RECALL_MISS = "candidate_recall_miss"
    BRAND_RETRIEVAL_MISS = "brand_retrieval_miss"
    POLICY_RETRIEVAL_MISS = "policy_retrieval_miss"
    PRICE_RETRIEVAL_MISS = "price_retrieval_miss"
    ROUTING_ERROR = "routing_error"
    PROMPT_INSTRUCTION_ERROR = "prompt_instruction_error"
    MODEL_REASONING_ERROR = "model_reasoning_error"
    SCHEMA_ERROR = "schema_error"
    AGGREGATION_ERROR = "aggregation_error"
    GUARDRAIL_ERROR = "guardrail_error"
    POLICY_GAP = "policy_gap"
    ANNOTATION_DISAGREEMENT = "annotation_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


class BadCaseSource(StringEnum):
    REVIEWER_DISAGREEMENT = "reviewer_disagreement"
    RESULT_REPORTED = "result_reported"
    EVALUATION_FAILURE = "evaluation_failure"
    RULE_CONFLICT = "rule_conflict"
    SCHEMA_FAILURE = "schema_failure"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DIVERGENCE = "model_divergence"


class AttributionStatus(StringEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


class BadCaseRecord(StrictModel):
    bad_case_id: str = Field(default_factory=lambda: f"BAD-{uuid4().hex[:12].upper()}")
    run_id: str
    case_id: str
    source: BadCaseSource
    suspected_failure_type: FailureType = FailureType.UNKNOWN
    confirmed_failure_type: Optional[FailureType] = None
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    attribution_note: str = Field(default="", max_length=4000)
    status: AttributionStatus = AttributionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReplayComparison(StrictModel):
    original_run_id: str
    replay_run_id: str
    first_differing_node: Optional[str]
    suggested_failure_type: FailureType


class BadCaseStore:
    def __init__(self, path: Path) -> None:
        self.store = JsonlStore(path, BadCaseRecord)

    def list(self) -> List[BadCaseRecord]:
        records = self.store.list()
        latest = {item.bad_case_id: item for item in records}
        return sorted(latest.values(), key=lambda item: item.created_at)

    def enqueue(self, record: BadCaseRecord) -> BadCaseRecord:
        self.store.append(record)
        return record

    def confirm(self, bad_case_id: str, failure_type: FailureType, confirmed_by: str, note: str = "") -> BadCaseRecord:
        current = next((item for item in self.list() if item.bad_case_id == bad_case_id), None)
        if current is None:
            raise KeyError(bad_case_id)
        confirmed = current.model_copy(update={"confirmed_failure_type": failure_type, "confirmed_by": confirmed_by, "confirmed_at": datetime.now(timezone.utc), "attribution_note": note, "status": AttributionStatus.CONFIRMED})
        confirmed = BadCaseRecord.model_validate(confirmed.model_dump())
        self.store.append(confirmed)
        return confirmed


_NODE_FAILURE = {
    "validate_input": FailureType.INPUT_MISSING,
    "normalize_and_extract_evidence": FailureType.EVIDENCE_EXTRACTION_ERROR,
    "recall_entities": FailureType.CANDIDATE_RECALL_MISS,
    "retrieve_policy": FailureType.POLICY_RETRIEVAL_MISS,
    "retrieve_brand_knowledge": FailureType.BRAND_RETRIEVAL_MISS,
    "retrieve_reference_price": FailureType.PRICE_RETRIEVAL_MISS,
    "evaluate_rules": FailureType.MODEL_REASONING_ERROR,
    "aggregate_decision": FailureType.AGGREGATION_ERROR,
    "apply_guardrail": FailureType.GUARDRAIL_ERROR,
}


def compare_replay(original: ExecutionTrace, replay: ExecutionTrace) -> ReplayComparison:
    if replay.parent_run_id != original.run_id:
        raise ValueError("replay trace must point to the original run_id")
    first = None
    for old, new in zip(original.nodes, replay.nodes):
        if old.node_name != new.node_name or old.output != new.output or old.status != new.status:
            first = new.node_name
            break
    if first is None and len(original.nodes) != len(replay.nodes):
        longer = replay.nodes if len(replay.nodes) > len(original.nodes) else original.nodes
        first = longer[min(len(original.nodes), len(replay.nodes))].node_name
    return ReplayComparison(
        original_run_id=original.run_id or "unknown",
        replay_run_id=replay.run_id or "unknown",
        first_differing_node=first,
        suggested_failure_type=_NODE_FAILURE.get(first or "", FailureType.UNKNOWN),
    )

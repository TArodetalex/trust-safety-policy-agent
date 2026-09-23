"""Human-review records that preserve agent and reviewer decisions separately."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field, model_validator

from trust_safety_agent.product_review import ProductReviewDecision, ReviewConfidence
from trust_safety_agent.schema import StrictModel, StringEnum


class ReviewWorkflow(StringEnum):
    GENERAL_CASE = "general_case"
    PRODUCT_IPR = "product_ipr"
    SHOP_IDENTITY = "shop_identity"


class ReviewerLabel(StringEnum):
    APPROVE = "approve"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class ReviewRecord(StrictModel):
    review_id: str = Field(pattern=r"^REV-[A-F0-9]{12}$")
    case_id: str = Field(min_length=1, max_length=100)
    workflow: ReviewWorkflow
    case_input: Dict[str, Any]
    agent_decision: ProductReviewDecision
    agent_confidence: ReviewConfidence
    agent_reason: str = Field(min_length=1, max_length=4000)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    policy_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    reviewer_label: Optional[ReviewerLabel] = None
    reviewer_note: str = Field(default="", max_length=4000)
    final_decision: Optional[ProductReviewDecision] = None
    add_to_golden_set: bool = False
    expected_policy: Optional[str] = Field(default=None, max_length=100)
    expected_evidence: List[str] = Field(default_factory=list)
    case_category: str = Field(default="", max_length=100)
    failure_type: str = Field(default="unknown", max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_review_state(self) -> "ReviewRecord":
        if self.reviewer_label is None:
            if self.final_decision is not None or self.reviewed_at is not None:
                raise ValueError("unreviewed records cannot have a final decision")
            if self.add_to_golden_set:
                raise ValueError("unreviewed records cannot enter the Golden Set")
            return self
        expected_final = {
            ReviewerLabel.APPROVE: ProductReviewDecision.APPROVE,
            ReviewerLabel.REJECT: ProductReviewDecision.REJECT,
            ReviewerLabel.UNCERTAIN: ProductReviewDecision.MANUAL_REVIEW,
        }[self.reviewer_label]
        if self.final_decision != expected_final:
            raise ValueError("final_decision must be derived from reviewer_label")
        if self.reviewed_at is None:
            raise ValueError("reviewed records require reviewed_at")
        if self.add_to_golden_set and self.reviewer_label == ReviewerLabel.UNCERTAIN:
            raise ValueError("uncertain reviews cannot enter the Golden Set")
        return self


class ReviewerFeedback(StrictModel):
    reviewer_label: ReviewerLabel
    reviewer_note: str = Field(min_length=1, max_length=4000)
    add_to_golden_set: bool = False
    expected_policy: Optional[str] = Field(default=None, max_length=100)
    expected_evidence: List[str] = Field(default_factory=list)
    case_category: str = Field(default="", max_length=100)
    failure_type: str = Field(default="unknown", max_length=100)

    @model_validator(mode="after")
    def validate_staging_request(self) -> "ReviewerFeedback":
        if self.add_to_golden_set and self.reviewer_label == ReviewerLabel.UNCERTAIN:
            raise ValueError("uncertain reviews cannot enter the Golden Set")
        return self


class ReviewerWorkspaceStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_records(self) -> List[ReviewRecord]:
        if not self.path.exists():
            return []
        records = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(ReviewRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid review record at line {line_number}: {exc}"
                ) from exc
        return records

    def _write(self, records: List[ReviewRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def enqueue(
        self,
        *,
        case_id: str,
        workflow: ReviewWorkflow,
        case_input: Dict[str, Any],
        agent_decision: ProductReviewDecision,
        agent_confidence: ReviewConfidence,
        agent_reason: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        policy_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> ReviewRecord:
        records = self.list_records()
        if any(
            record.case_id == case_id and record.reviewer_label is None
            for record in records
        ):
            raise ValueError(f"case {case_id} already has a pending review")
        record = ReviewRecord(
            review_id=f"REV-{uuid4().hex[:12].upper()}",
            case_id=case_id,
            workflow=workflow,
            case_input=case_input,
            agent_decision=agent_decision,
            agent_confidence=agent_confidence,
            agent_reason=agent_reason,
            evidence=evidence or [],
            policy_evidence=policy_evidence or [],
        )
        records.append(record)
        self._write(records)
        return record

    def complete(self, review_id: str, feedback: ReviewerFeedback) -> ReviewRecord:
        records = self.list_records()
        for index, record in enumerate(records):
            if record.review_id != review_id:
                continue
            if record.reviewer_label is not None:
                raise ValueError(f"review {review_id} is already complete")
            final_decision = {
                ReviewerLabel.APPROVE: ProductReviewDecision.APPROVE,
                ReviewerLabel.REJECT: ProductReviewDecision.REJECT,
                ReviewerLabel.UNCERTAIN: ProductReviewDecision.MANUAL_REVIEW,
            }[feedback.reviewer_label]
            completed = record.model_copy(
                update={
                    "reviewer_label": feedback.reviewer_label,
                    "reviewer_note": feedback.reviewer_note,
                    "final_decision": final_decision,
                    "add_to_golden_set": feedback.add_to_golden_set,
                    "expected_policy": feedback.expected_policy,
                    "expected_evidence": feedback.expected_evidence,
                    "case_category": feedback.case_category,
                    "failure_type": feedback.failure_type,
                    "reviewed_at": datetime.now(timezone.utc),
                }
            )
            completed = ReviewRecord.model_validate(completed.model_dump())
            records[index] = completed
            self._write(records)
            return completed
        raise KeyError(f"review {review_id} was not found")

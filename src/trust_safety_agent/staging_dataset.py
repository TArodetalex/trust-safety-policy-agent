"""Candidate dataset storage and explicit promotion without mutating v4."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.reviewer_workspace import ReviewRecord, ReviewerLabel
from trust_safety_agent.schema import StrictModel, StringEnum


class StagingStatus(StringEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"


class StagingGoldenCase(StrictModel):
    candidate_id: str = Field(pattern=r"^STG-[A-F0-9]{12}$")
    review_id: str
    case_id: str
    case_input: Dict[str, Any]
    expected_decision: str = Field(pattern=r"^(approve|reject)$")
    expected_policy: Optional[str] = None
    expected_evidence: List[str] = Field(default_factory=list)
    case_category: str
    failure_type: str
    reviewer_note: str
    dataset_version: str = "staging-v1.0.0"
    status: StagingStatus = StagingStatus.CANDIDATE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: Optional[datetime] = None


class PromotionManifest(StrictModel):
    dataset_version: str
    source_dataset: str = "staging-v1.0.0"
    record_count: int = Field(ge=0)
    data_file: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StagingDatasetStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_records(self) -> List[StagingGoldenCase]:
        if not self.path.exists():
            return []
        return [
            StagingGoldenCase.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write(self, records: List[StagingGoldenCase]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def add_review(self, review: ReviewRecord) -> StagingGoldenCase:
        if not review.add_to_golden_set:
            raise ValueError("review is not marked for the staging dataset")
        if review.reviewer_label not in {ReviewerLabel.APPROVE, ReviewerLabel.REJECT}:
            raise ValueError("only decisive reviewer labels may enter staging")
        records = self.list_records()
        if any(item.review_id == review.review_id for item in records):
            raise ValueError(f"review {review.review_id} already exists in staging")
        candidate = StagingGoldenCase(
            candidate_id=f"STG-{uuid4().hex[:12].upper()}",
            review_id=review.review_id,
            case_id=review.case_id,
            case_input=review.case_input,
            expected_decision=review.reviewer_label.value,
            expected_policy=review.expected_policy,
            expected_evidence=review.expected_evidence,
            case_category=review.case_category or review.workflow.value,
            failure_type=review.failure_type,
            reviewer_note=review.reviewer_note,
        )
        records.append(candidate)
        self._write(records)
        return candidate

    def promote(
        self,
        output_directory: Path,
        dataset_version: str = "v5.0.0",
    ) -> PromotionManifest:
        if not dataset_version.startswith("v5."):
            raise ValueError("staging promotion currently targets Golden Set v5")
        records = self.list_records()
        candidates = [item for item in records if item.status == StagingStatus.CANDIDATE]
        if not candidates:
            raise ValueError("staging dataset has no candidates to promote")
        output_directory.mkdir(parents=True, exist_ok=True)
        data_path = output_directory / "golden_set_v5.jsonl"
        content = "".join(
            item.model_copy(update={"dataset_version": dataset_version}).model_dump_json()
            + "\n"
            for item in candidates
        )
        data_path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        manifest = PromotionManifest(
            dataset_version=dataset_version,
            record_count=len(candidates),
            data_file=data_path.name,
            sha256=digest,
        )
        (output_directory / "golden_set_v5_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        now = datetime.now(timezone.utc)
        promoted_ids = {item.candidate_id for item in candidates}
        self._write(
            [
                item.model_copy(
                    update={"status": StagingStatus.PROMOTED, "promoted_at": now}
                )
                if item.candidate_id in promoted_ids
                else item
                for item in records
            ]
        )
        return manifest

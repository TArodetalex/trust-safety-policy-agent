"""Controlled model-only experiments and versioned publication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from pydantic import Field, model_validator

from trust_safety_agent.case_store import JsonlStore
from trust_safety_agent.schema import StrictModel, StringEnum


class ReleaseStatus(StringEnum):
    DRAFT = "draft"
    EVALUATION = "evaluation"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class WorkflowConfiguration(StrictModel):
    config_id: str = Field(default_factory=lambda: f"CFG-{uuid4().hex[:10].upper()}")
    version: int = Field(default=1, ge=1)
    workflow: str
    model_id: str
    dataset_version: str
    workflow_version: str
    prompt_version: str
    skill_version: str
    policy_index_version: str
    brand_index_version: str
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=1500, ge=1)
    output_schema: str = "phase9-review-v1"
    price_fixture_version: str = "v1.0.0"
    status: ReleaseStatus = ReleaseStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def controlled_fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"config_id", "version", "model_id", "status", "created_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExperimentMetrics(StrictModel):
    accuracy: float = Field(ge=0, le=1)
    false_approve: int = Field(ge=0)
    false_reject: int = Field(ge=0)
    manual_review_rate: float = Field(ge=0, le=1)
    schema_valid_rate: float = Field(ge=0, le=1)
    exact_match_rate: float = Field(ge=0, le=1)
    conflict_rate: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)
    token_usage: int = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    bad_case_count: int = Field(ge=0)
    label_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class ModelExperimentResult(StrictModel):
    experiment_id: str = Field(default_factory=lambda: f"EXP-{uuid4().hex[:10].upper()}")
    baseline: WorkflowConfiguration
    candidates: List[WorkflowConfiguration] = Field(min_length=1, max_length=2)
    metrics_by_model: Dict[str, ExperimentMetrics]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_controlled_variables(self) -> "ModelExperimentResult":
        fingerprint = self.baseline.controlled_fingerprint()
        configs = [self.baseline, *self.candidates]
        if any(item.controlled_fingerprint() != fingerprint for item in configs):
            raise ValueError("model experiments may change only model_id")
        if len({item.model_id for item in configs}) != len(configs):
            raise ValueError("experiment model IDs must be unique")
        if set(self.metrics_by_model) != {item.model_id for item in configs}:
            raise ValueError("metrics must exist for every experiment model")
        return self


class WorkflowConfigurationStore:
    def __init__(self, root: Path) -> None:
        self.store = JsonlStore(root / "workflow_configurations.jsonl", WorkflowConfiguration)

    def save(self, config: WorkflowConfiguration) -> WorkflowConfiguration:
        self.store.append(config)
        return config

    def publish(self, config: WorkflowConfiguration) -> WorkflowConfiguration:
        prior = self.store.list()
        version = max((item.version for item in prior if item.workflow == config.workflow), default=0) + 1
        published = config.model_copy(update={"config_id": f"CFG-{uuid4().hex[:10].upper()}", "version": version, "status": ReleaseStatus.PUBLISHED, "created_at": datetime.now(timezone.utc)})
        published = WorkflowConfiguration.model_validate(published.model_dump())
        return self.save(published)

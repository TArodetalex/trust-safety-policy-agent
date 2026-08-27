"""Versioned data contracts for cases, decisions, and evaluation runs."""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


SCHEMA_VERSION = "1.2.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StringEnum(str, Enum):
    pass


class ContentType(StringEnum):
    VIDEO = "video"
    PRODUCT = "product"
    SHOP_NAME = "shop_name"
    SHOP_AVATAR = "shop_avatar"
    QUERY = "query"


class HumanDecision(StringEnum):
    APPROVE = "approve"
    REJECT = "reject"


class AgentDecisionLabel(StringEnum):
    APPROVE = "approve"
    REJECT = "reject"
    NEED_REVIEW = "need_review"


class PolicyLabel(StringEnum):
    COUNTERFEIT = "counterfeit"
    KNOCKOFF = "knockoff"
    TRADEMARK_MISUSE = "trademark_misuse"
    SHOP_IMPERSONATION = "shop_impersonation"
    RISKY_QUERY = "risky_query"


class ExemptionType(StringEnum):
    NONE = "none"
    COMPATIBILITY = "compatibility"
    SECOND_HAND = "second_hand"
    MEANINGFUL_WORD = "meaningful_word"
    INCIDENTAL_EXPOSURE = "incidental_exposure"
    CO_BRAND = "co_brand"


class RiskLevel(StringEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DatasetSplit(StringEnum):
    DEVELOPMENT = "development"
    REGRESSION = "regression"
    BLIND = "blind"


class ExpectedRoute(StringEnum):
    AUTO_DECIDE = "auto_decide"
    HUMAN_REVIEW = "human_review"


class ErrorType(StringEnum):
    NONE = "none"
    FALSE_REJECT = "false_reject"
    FALSE_APPROVE = "false_approve"
    WRONG_POLICY = "wrong_policy"
    RETRIEVAL_MISS = "retrieval_miss"
    INVALID_OUTPUT = "invalid_output"
    UNEXPECTED_REVIEW = "unexpected_review"
    UNEXPECTED_AUTO_DECISION = "unexpected_auto_decision"
    UNSAFE_AUTO_DECISION = "unsafe_auto_decision"


class PolicyChunk(StrictModel):
    """One retrievable policy unit with stable provenance metadata."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    chunk_id: str = Field(pattern=r"^PCH-[a-f0-9]{12}$")
    policy_id: str = Field(pattern=r"^POL-[A-Z]{2}-\d{3}$")
    title: Annotated[str, Field(min_length=1, max_length=200)]
    heading_path: List[str] = Field(min_length=1)
    source: Annotated[str, Field(min_length=1, max_length=500)]
    policy_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    chunk_index: int = Field(ge=0)
    content: Annotated[str, Field(min_length=20, max_length=4000)]
    policy_label: Optional[PolicyLabel] = None
    exemption_type: ExemptionType = ExemptionType.NONE

    @model_validator(mode="after")
    def validate_taxonomy(self) -> PolicyChunk:
        if self.policy_label is not None and self.exemption_type != ExemptionType.NONE:
            raise ValueError("a chunk cannot be both a violation and an exemption")
        return self


class RetrievalHit(StrictModel):
    chunk: PolicyChunk
    score: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)


class GoldenCase(StrictModel):
    """Human-authored ground truth. Agent-generated fields do not belong here."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    dataset_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    case_id: str = Field(pattern=r"^C\d{3}$")
    content_type: ContentType
    input_text: Annotated[str, Field(min_length=1, max_length=2000)]
    image_url: Optional[HttpUrl] = None
    image_path: Optional[
        Annotated[str, Field(pattern=r"^data/golden_set/assets/.+\.(png|jpe?g|webp)$")]
    ] = None
    human_decision: HumanDecision
    expected_policy: Optional[PolicyLabel] = None
    expected_exemption: ExemptionType = ExemptionType.NONE
    risk_level: RiskLevel
    is_boundary_case: bool
    human_reason: Annotated[str, Field(min_length=8, max_length=1000)]
    source: Annotated[str, Field(min_length=1, max_length=100)] = "synthetic"
    split: DatasetSplit = DatasetSplit.REGRESSION
    expected_route: ExpectedRoute = ExpectedRoute.AUTO_DECIDE
    tags: List[Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_ground_truth(self) -> GoldenCase:
        if self.image_url and self.image_path:
            raise ValueError("case cannot declare both image_url and image_path")
        if self.image_path:
            path = PurePosixPath(self.image_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("image_path must be a safe project-relative path")
        if self.human_decision == HumanDecision.REJECT:
            if self.expected_policy is None:
                raise ValueError("reject cases require expected_policy")
            if self.expected_exemption != ExemptionType.NONE:
                raise ValueError("reject cases cannot declare an exemption")
        if self.human_decision == HumanDecision.APPROVE:
            if self.expected_policy is not None:
                raise ValueError("approve cases cannot declare expected_policy")
            if (
                self.expected_exemption != ExemptionType.NONE
                and not self.is_boundary_case
            ):
                raise ValueError("exemption cases must be marked as boundary cases")
        return self


class MatchedPolicyEvidence(StrictModel):
    policy_id: str
    chunk_id: str
    quote: Annotated[str, Field(min_length=1)]
    retrieval_score: Optional[float] = Field(default=None, ge=0, le=1)
    exemption_type: ExemptionType = ExemptionType.NONE


class AgentDecision(StrictModel):
    """Normalized output contract for one policy judgment."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    case_id: str = Field(pattern=r"^C\d{3}$")
    decision: AgentDecisionLabel
    policy_label: Optional[PolicyLabel] = None
    matched_policy: List[MatchedPolicyEvidence] = Field(default_factory=list)
    reason: Annotated[str, Field(min_length=1, max_length=2000)]
    recommended_action: Annotated[str, Field(min_length=1, max_length=500)]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_decision(self) -> AgentDecision:
        if self.decision == AgentDecisionLabel.REJECT and self.policy_label is None:
            raise ValueError("reject decisions require policy_label")
        if self.decision != AgentDecisionLabel.REJECT and self.policy_label is not None:
            raise ValueError("only reject decisions may declare policy_label")
        if self.decision == AgentDecisionLabel.REJECT and not self.matched_policy:
            raise ValueError("reject decisions require matched policy evidence")
        return self


class EvaluationRecord(StrictModel):
    """One immutable comparison between human truth and an agent decision."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    eval_run_id: str
    case_id: str = Field(pattern=r"^C\d{3}$")
    prompt_version: str
    retrieval_version: str
    human_decision: HumanDecision
    expected_policy: Optional[PolicyLabel] = None
    agent_decision: AgentDecisionLabel
    agent_policy: Optional[PolicyLabel] = None
    is_decision_correct: bool
    is_policy_correct: Optional[bool] = None
    expected_route: ExpectedRoute = ExpectedRoute.AUTO_DECIDE
    is_route_correct: bool = True
    confidence: float = Field(default=0, ge=0, le=1)
    content_type: Optional[ContentType] = None
    risk_level: Optional[RiskLevel] = None
    split: Optional[DatasetSplit] = None
    is_boundary_case: bool = False
    tags: List[str] = Field(default_factory=list)
    agent_reason: str = ""
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    error_type: ErrorType = ErrorType.NONE
    latency_ms: int = Field(ge=0)


class EvaluationMetrics(StrictModel):
    eval_run_id: str
    total_cases: int = Field(ge=0)
    auto_decided_cases: int = Field(ge=0)
    need_review_cases: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    auto_accuracy: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    route_accuracy: float = Field(ge=0, le=1)
    policy_accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_reject_rate: float = Field(ge=0, le=1)
    false_approve_rate: float = Field(ge=0, le=1)
    review_rate: float = Field(ge=0, le=1)
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    error_counts: Dict[str, int] = Field(default_factory=dict)


class EvaluationReport(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    eval_run_id: str
    generated_at: datetime
    dataset_version: str
    dataset_fingerprint: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    engine: str
    prompt_version: str
    retrieval_version: str
    metrics: EvaluationMetrics
    slices: Dict[str, EvaluationMetrics] = Field(default_factory=dict)
    gates: Dict[str, float] = Field(default_factory=dict)
    failed_gates: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed_gates


class ShadowEvaluationRecord(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    eval_run_id: str
    case_id: str = Field(pattern=r"^C\d{3}$")
    requested_model: str
    response_model: Optional[str] = None
    provider: Optional[str] = None
    request_id: Optional[str] = None
    status_code: Optional[int] = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost: Optional[float] = Field(default=None, ge=0)
    schema_valid: bool
    production_decision: AgentDecisionLabel
    shadow_decision: AgentDecisionLabel
    shadow_policy: Optional[PolicyLabel] = None
    decision_correct: bool
    policy_correct: Optional[bool] = None
    confidence: float = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    reason: str
    split: DatasetSplit
    tags: List[str] = Field(default_factory=list)


class ShadowEvaluationSummary(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    eval_run_id: str
    requested_model: str
    total_cases: int = Field(ge=0)
    schema_valid_cases: int = Field(ge=0)
    schema_success_rate: float = Field(ge=0, le=1)
    decision_accuracy: float = Field(ge=0, le=1)
    policy_accuracy: float = Field(ge=0, le=1)
    false_approve_rate: float = Field(ge=0, le=1)
    false_reject_rate: float = Field(ge=0, le=1)
    total_tokens: int = Field(ge=0)
    total_cost: float = Field(ge=0)
    provider_counts: Dict[str, int] = Field(default_factory=dict)
    response_model_counts: Dict[str, int] = Field(default_factory=dict)
    gates: Dict[str, float] = Field(default_factory=dict)
    failed_gates: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed_gates

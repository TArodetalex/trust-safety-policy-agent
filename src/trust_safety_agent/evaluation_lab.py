"""Read and compare reproducible evaluation runs for the product UI."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from pydantic import Field

from trust_safety_agent.schema import EvaluationReport, StrictModel


def run_offline_rules_experiment(
    *,
    run_id: str,
    dataset_path: Path,
    output_root: Path,
    store,
) -> EvaluationReport:
    """Run one reproducible offline experiment without shelling out from the UI."""
    from trust_safety_agent.adjudicator import PolicyAdjudicator
    from trust_safety_agent.dataset import load_cases, validate_dataset
    from trust_safety_agent.evaluation import evaluate_cases, write_evaluation_artifacts

    normalized = run_id.strip().upper()
    if not normalized or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in normalized):
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores")
    output_directory = output_root / normalized
    if output_directory.exists():
        raise ValueError(f"evaluation run {normalized} already exists")
    cases = load_cases(dataset_path)
    errors = validate_dataset(cases)
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(errors[:5]))
    agent = PolicyAdjudicator(store)
    report, records = evaluate_cases(
        cases,
        lambda case: agent.adjudicate(
            case_id=case.case_id,
            content_type=case.content_type,
            input_text=case.input_text,
        ),
        engine="rules",
        prompt_version="rules-v1",
        retrieval_version=store.collection_name,
        eval_run_id=normalized,
    )
    write_evaluation_artifacts(report, records, output_directory)
    return report


class EvaluationTelemetry(StrictModel):
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0, ge=0)
    average_latency_ms: float = Field(default=0, ge=0)
    p95_latency_ms: int = Field(default=0, ge=0)
    provider_counts: Dict[str, int] = Field(default_factory=dict)
    response_model_counts: Dict[str, int] = Field(default_factory=dict)


class EvaluationRunSummary(StrictModel):
    run_id: str
    path: str
    report: EvaluationReport
    telemetry: EvaluationTelemetry


class MetricDelta(StrictModel):
    metric: str
    baseline: float
    candidate: float
    delta: float


class EvaluationComparison(StrictModel):
    baseline_run_id: str
    candidate_run_id: str
    metric_deltas: List[MetricDelta]
    newly_failed_gates: List[str]
    resolved_failed_gates: List[str]
    token_delta: int
    cost_delta: float
    latency_delta_ms: float


_METRICS = (
    "accuracy",
    "auto_accuracy",
    "coverage",
    "route_accuracy",
    "policy_accuracy",
    "false_approve_rate",
    "false_reject_rate",
    "review_rate",
)


def _percentile_95(values: List[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return ordered[index]


class EvaluationLabStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _telemetry(directory: Path) -> EvaluationTelemetry:
        shadow_report = directory / "shadow_report.json"
        if shadow_report.exists():
            payload = json.loads(shadow_report.read_text(encoding="utf-8"))
            return EvaluationTelemetry(
                total_tokens=int(payload.get("total_tokens") or 0),
                total_cost=float(payload.get("total_cost") or 0),
                provider_counts=payload.get("provider_counts") or {},
                response_model_counts=payload.get("response_model_counts") or {},
            )
        records_path = directory / "records.jsonl"
        if not records_path.exists():
            return EvaluationTelemetry()
        latencies = []
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                latencies.append(int(payload.get("latency_ms") or 0))
        return EvaluationTelemetry(
            average_latency_ms=mean(latencies) if latencies else 0,
            p95_latency_ms=_percentile_95(latencies),
        )

    def list_runs(self) -> List[EvaluationRunSummary]:
        if not self.root.exists():
            return []
        runs = []
        for report_path in self.root.glob("*/report.json"):
            try:
                report = EvaluationReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
                runs.append(
                    EvaluationRunSummary(
                        run_id=report.eval_run_id,
                        path=str(report_path.parent),
                        report=report,
                        telemetry=self._telemetry(report_path.parent),
                    )
                )
            except (ValueError, json.JSONDecodeError):
                continue
        return sorted(runs, key=lambda item: item.report.generated_at)

    def get(self, run_id: str) -> Optional[EvaluationRunSummary]:
        return next((item for item in self.list_runs() if item.run_id == run_id), None)

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> EvaluationComparison:
        baseline = self.get(baseline_run_id)
        candidate = self.get(candidate_run_id)
        if baseline is None or candidate is None:
            raise KeyError("evaluation run was not found")
        deltas = [
            MetricDelta(
                metric=name,
                baseline=float(getattr(baseline.report.metrics, name)),
                candidate=float(getattr(candidate.report.metrics, name)),
                delta=round(
                    float(getattr(candidate.report.metrics, name))
                    - float(getattr(baseline.report.metrics, name)),
                    10,
                ),
            )
            for name in _METRICS
        ]
        baseline_failures = set(baseline.report.failed_gates)
        candidate_failures = set(candidate.report.failed_gates)
        return EvaluationComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            metric_deltas=deltas,
            newly_failed_gates=sorted(candidate_failures - baseline_failures),
            resolved_failed_gates=sorted(baseline_failures - candidate_failures),
            token_delta=candidate.telemetry.total_tokens - baseline.telemetry.total_tokens,
            cost_delta=round(
                candidate.telemetry.total_cost - baseline.telemetry.total_cost,
                10,
            ),
            latency_delta_ms=round(
                candidate.telemetry.average_latency_ms
                - baseline.telemetry.average_latency_ms,
                3,
            ),
        )

from datetime import datetime, timezone
from pathlib import Path

from trust_safety_agent.evaluation_lab import EvaluationLabStore
from trust_safety_agent.schema import EvaluationMetrics, EvaluationReport


def write_report(root: Path, run_id: str, coverage: float, failures: list[str]):
    directory = root / run_id
    directory.mkdir(parents=True)
    metrics = EvaluationMetrics(
        eval_run_id=run_id,
        total_cases=10,
        auto_decided_cases=round(coverage * 10),
        need_review_cases=10 - round(coverage * 10),
        accuracy=coverage,
        auto_accuracy=1,
        coverage=coverage,
        route_accuracy=coverage,
        policy_accuracy=1,
        precision=1,
        recall=coverage,
        f1=coverage,
        false_reject_rate=0,
        false_approve_rate=0,
        review_rate=1 - coverage,
    )
    report = EvaluationReport(
        eval_run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        dataset_version="v4.0.0",
        dataset_fingerprint="sha256:" + "a" * 64,
        engine="rules",
        prompt_version="rules-v1",
        retrieval_version="policy-v1",
        metrics=metrics,
        failed_gates=failures,
    )
    (directory / "report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / "records.jsonl").write_text(
        '{"latency_ms": 10}\n{"latency_ms": 30}\n', encoding="utf-8"
    )


def test_evaluation_lab_indexes_and_compares_runs(tmp_path: Path):
    write_report(tmp_path, "EV-A", 0.6, ["coverage failed"])
    write_report(tmp_path, "EV-B", 0.8, [])
    lab = EvaluationLabStore(tmp_path)

    assert [run.run_id for run in lab.list_runs()] == ["EV-A", "EV-B"]
    comparison = lab.compare("EV-A", "EV-B")
    coverage = next(item for item in comparison.metric_deltas if item.metric == "coverage")
    assert coverage.delta == 0.2
    assert comparison.resolved_failed_gates == ["coverage failed"]
    assert comparison.latency_delta_ms == 0

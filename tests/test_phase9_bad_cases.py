from datetime import datetime, timezone
from pathlib import Path

from trust_safety_agent.bad_cases import BadCaseRecord, BadCaseSource, BadCaseStore, FailureType, compare_replay
from trust_safety_agent.trace import ExecutionTrace, TraceNode, TraceNodeStatus


def make_trace(run_id: str, output: dict, parent_run_id: str | None = None) -> ExecutionTrace:
    return ExecutionTrace(trace_id=f"TRC-{run_id[-12:]}", run_id=run_id, parent_run_id=parent_run_id, case_id="P-1", case_revision=1, workflow="product_ipr", nodes=[TraceNode(node_name="recall_entities", node_id="recall_entities", status=TraceNodeStatus.SUCCESS, latency_ms=1, input_summary={}, output=output, started_at=datetime.now(timezone.utc), ended_at=datetime.now(timezone.utc))])


def test_bad_case_requires_human_confirmation_and_replay_finds_first_difference(tmp_path: Path) -> None:
    store = BadCaseStore(tmp_path / "bad.jsonl")
    record = store.enqueue(BadCaseRecord(run_id="RUN-000000000001", case_id="P-1", source=BadCaseSource.REVIEWER_DISAGREEMENT, suspected_failure_type=FailureType.CANDIDATE_RECALL_MISS))
    assert record.confirmed_failure_type is None
    confirmed = store.confirm(record.bad_case_id, FailureType.CANDIDATE_RECALL_MISS, "ops-user", "brand was missed")
    assert confirmed.confirmed_by == "ops-user"
    assert store.list()[0].confirmed_failure_type == FailureType.CANDIDATE_RECALL_MISS

    original = make_trace("RUN-000000000001", {"detected_brand": None})
    replay = make_trace("RUN-000000000002", {"detected_brand": "Nike"}, parent_run_id=original.run_id)
    comparison = compare_replay(original, replay)
    assert comparison.first_differing_node == "recall_entities"
    assert comparison.suggested_failure_type == FailureType.CANDIDATE_RECALL_MISS

from pathlib import Path

from trust_safety_agent.trace import (
    TraceNode,
    TraceNodeStatus,
    TraceStore,
    TraceTokenUsage,
    build_trace,
    sanitize_sensitive,
)


def test_sensitive_fields_and_values_are_redacted_recursively() -> None:
    sanitized = sanitize_sensitive(
        {
            "api_key": "sk-test-1234567890",
            "nested": {
                "password": "secret-value",
                "message": "Bearer abcdefghijklmnop",
                "authorization_status": "authorized",
            },
        }
    )

    assert sanitized["api_key"] == "<redacted>"
    assert sanitized["nested"]["password"] == "<redacted>"
    assert sanitized["nested"]["message"] == "<redacted>"
    assert sanitized["nested"]["authorization_status"] == "authorized"


def test_trace_is_structured_persisted_and_filterable(tmp_path: Path) -> None:
    trace = build_trace(
        case_id="PR-300",
        workflow="product_ipr",
        nodes=[
            TraceNode(
                node_name="retrieval",
                input_summary={"query": "Gucci replica"},
                output={"hits": ["POL-CF-001"]},
                status=TraceNodeStatus.SUCCESS,
                latency_ms=12,
                token_usage=TraceTokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
                confidence=0.9,
            )
        ],
        final_output={"decision": "reject", "api_key": "never-store-this"},
    )
    store = TraceStore(tmp_path / "traces.jsonl")
    store.append(trace)

    loaded = store.list_traces(case_id="PR-300")
    assert loaded[0].nodes[0].node_name == "retrieval"
    assert loaded[0].final_output["api_key"] == "<redacted>"
    assert store.list_traces(case_id="missing") == []


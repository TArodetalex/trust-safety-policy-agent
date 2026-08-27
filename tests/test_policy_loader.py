from pathlib import Path

import pytest

from trust_safety_agent.policy_loader import load_policy_file, load_policy_text
from trust_safety_agent.schema import ExemptionType, PolicyLabel


POLICY_PATH = Path(__file__).parents[1] / "data/policies/mock_policy_v1.md"


def test_policy_loader_preserves_taxonomy_and_provenance() -> None:
    chunks = load_policy_file(POLICY_PATH)

    assert len(chunks) >= 11
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.policy_label for chunk in chunks if chunk.policy_label} == set(
        PolicyLabel
    )
    assert {
        chunk.exemption_type
        for chunk in chunks
        if chunk.exemption_type != ExemptionType.NONE
    } == set(ExemptionType) - {ExemptionType.NONE}
    assert all(chunk.source == "mock_policy_v1.md" for chunk in chunks)
    assert all(chunk.policy_version == "v1.0.0" for chunk in chunks)


def test_policy_loader_creates_stable_chunk_ids() -> None:
    first = load_policy_file(POLICY_PATH)
    second = load_policy_file(POLICY_PATH)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_policy_loader_rejects_invalid_chunk_settings() -> None:
    with pytest.raises(ValueError, match="at least 200"):
        load_policy_text(
            "## POL-CF-001 Counterfeit\n\nCounterfeit policy body.",
            source="test.md",
            max_chars=100,
        )

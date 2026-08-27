from pathlib import Path

from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.vector_store import PolicyVectorStore


POLICY_PATH = Path(__file__).parents[1] / "data/policies/mock_policy_v1.md"


def test_chroma_index_round_trip(tmp_path: Path) -> None:
    chunks = load_policy_file(POLICY_PATH)
    store = PolicyVectorStore(tmp_path / "chroma")

    indexed = store.index(chunks)

    assert indexed == len(chunks)
    assert store.count() == len(chunks)
    assert [chunk.chunk_id for chunk in store.list_chunks()] == [
        chunk.chunk_id
        for chunk in sorted(
            chunks, key=lambda item: (item.policy_id, item.title, item.chunk_index)
        )
    ]


def test_retrieval_returns_relevant_policy_first(tmp_path: Path) -> None:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(POLICY_PATH))

    cases = {
        "counterfeit fake replica branded goods": "POL-CF-001",
        "unbranded sandals with the same protected design": "POL-KO-001",
        "generic shirt printed with a protected trademark logo": "POL-TM-001",
        "seller claims to be an authorized official brand store": "POL-SI-001",
        "case compatible with iPhone": "POL-EX-001",
        "fake goods pass authenticity checks": "POL-RQ-001",
    }

    for query, expected_policy_id in cases.items():
        hits = store.retrieve(query, top_k=3)
        assert hits
        assert expected_policy_id in {hit.chunk.policy_id for hit in hits[:2]}
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))

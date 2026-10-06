from pathlib import Path

from trust_safety_agent.brand_knowledge import BrandKnowledgeStore, load_brand_knowledge
from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_KNOWLEDGE_PATH, DEFAULT_BRAND_LIBRARY_PATH
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.vector_store import PolicyVectorStore
from trust_safety_agent.config import DEFAULT_POLICY_PATH


def test_public_brand_sync_contains_50_plus_synthetic_fixtures() -> None:
    library = ControlledBrandLibrary.from_csv_and_knowledge(DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_BRAND_KNOWLEDGE_PATH)
    assert len([item for item in library.brands if item.source_group == "public"]) == 50
    assert len([item for item in library.brands if item.source_group == "synthetic"]) == 6
    assert library.get("iphone").brand_name == "Apple"
    assert library.get("Levis").brand_name == "Levi's"


def test_policy_and_brand_indexes_are_independent(tmp_path: Path) -> None:
    policy_store = PolicyVectorStore(tmp_path, collection_name="policy_chunks_v2")
    brand_store = BrandKnowledgeStore(tmp_path)
    policy_store.index(load_policy_file(DEFAULT_POLICY_PATH))
    brand_store.index(load_brand_knowledge(DEFAULT_BRAND_KNOWLEDGE_PATH))
    assert policy_store.collection_name != brand_store.collection_name
    assert brand_store.count() == 50
    brand_hits = brand_store.retrieve("Air Force sneaker Nike")
    assert brand_hits[0].chunk.brand_name == "Nike"
    assert all(hit.chunk.chunk_id.startswith("BRAND-") for hit in brand_hits)
    assert all(hit.chunk.chunk_id.startswith("PCH-") for hit in policy_store.retrieve("compatibility exemption"))

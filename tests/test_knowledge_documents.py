from pathlib import Path

from trust_safety_agent.knowledge_documents import KnowledgeDocumentStore, KnowledgeKind


def test_policy_document_preserves_raw_source_and_builds_chunks(tmp_path: Path) -> None:
    store = KnowledgeDocumentStore(tmp_path)
    content = """# 合成政策\n\nVersion: `v1.0.0`\n\n## POL-TST-001 测试规则\n\n命中明确证据时拒绝，证据不足时转人工。""".encode("utf-8")
    document = store.save("policy.md", content, KnowledgeKind.POLICY)
    assert store.read_text(document).startswith("# 合成政策")
    chunks = store.policy_chunks(document)
    assert len(chunks) == 1
    assert chunks[0].policy_id == "POL-TST-001"


def test_unstructured_policy_document_gets_a_stable_upload_wrapper(tmp_path: Path) -> None:
    store = KnowledgeDocumentStore(tmp_path)
    document = store.save("notes.txt", "这是公开、合成的政策说明。证据不足时转人工。".encode("utf-8"), KnowledgeKind.POLICY)
    chunks = store.policy_chunks(document)
    assert chunks
    assert chunks[0].policy_id.startswith("POL-UPL-")

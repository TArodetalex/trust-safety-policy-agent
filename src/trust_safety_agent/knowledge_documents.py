"""Raw knowledge document storage and deterministic Policy RAG ingestion."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.case_store import JsonlStore
from trust_safety_agent.policy_loader import load_policy_text
from trust_safety_agent.schema import PolicyChunk, StrictModel, StringEnum


class KnowledgeKind(StringEnum):
    POLICY = "policy"
    BRAND = "brand"


class KnowledgeDocument(StrictModel):
    document_id: str = Field(default_factory=lambda: f"DOC-{uuid4().hex[:10].upper()}")
    name: str = Field(min_length=1, max_length=200)
    kind: KnowledgeKind
    source_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1)
    status: str = Field(default="stored", pattern=r"^(stored|indexed|needs_structure)$")
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_at: Optional[datetime] = None


class KnowledgeDocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = root / "raw"
        self.records = JsonlStore(root / "documents.jsonl", KnowledgeDocument)

    def list(self, kind: Optional[KnowledgeKind] = None) -> List[KnowledgeDocument]:
        latest = {}
        for item in self.records.list():
            latest[item.document_id] = item
        records = list(latest.values())
        if kind is not None:
            records = [item for item in records if item.kind == kind]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def save(self, name: str, content: bytes, kind: KnowledgeKind) -> KnowledgeDocument:
        if not content:
            raise ValueError("知识文档不能为空")
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("单个知识文档不能超过 2 MB")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("当前仅支持 UTF-8 编码的 Markdown 或文本文件") from exc
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip("-.") or "knowledge.md"
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("raw") / f"{digest[:12]}-{safe_name}"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        document = KnowledgeDocument(
            name=Path(name).name,
            kind=kind,
            source_path=relative.as_posix(),
            sha256=digest,
            size_bytes=len(content),
        )
        self.records.append(document)
        return document

    def read_text(self, document: KnowledgeDocument) -> str:
        return (self.root / document.source_path).read_text(encoding="utf-8")

    def policy_chunks(self, document: KnowledgeDocument) -> List[PolicyChunk]:
        if document.kind != KnowledgeKind.POLICY:
            raise ValueError("只有政策文档可以写入 Policy RAG")
        text = self.read_text(document)
        chunks = load_policy_text(text, source=document.name)
        if chunks:
            return chunks
        digest_number = int(document.sha256[:8], 16) % 1000
        wrapped = (
            "# Uploaded Synthetic Policy Reference\n\n"
            "Version: `v1.0.0`\n\n"
            f"## POL-UPL-{digest_number:03d} 上传政策参考\n\n{text}"
        )
        return load_policy_text(wrapped, source=document.name)

    def mark_indexed(self, document: KnowledgeDocument, chunk_count: int) -> KnowledgeDocument:
        updated = document.model_copy(update={
            "status": "indexed",
            "chunk_count": chunk_count,
            "indexed_at": datetime.now(timezone.utc),
        })
        self.records.append(KnowledgeDocument.model_validate(updated.model_dump()))
        return updated

"""Independent brand/product knowledge loader and Chroma collection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import chromadb
from pydantic import Field

from trust_safety_agent.embeddings import HashingEmbedder
from trust_safety_agent.schema import StrictModel


_VERSION = re.compile(r"^Version:\s+`?(?P<version>v[\w.-]+)`?", re.MULTILINE)
_SECTION = re.compile(r"^##\s+(?P<id>BRAND-\d{3})\s+(?P<name>.+)$", re.MULTILINE)


class BrandKnowledgeChunk(StrictModel):
    chunk_id: str = Field(pattern=r"^BRAND-\d{3}$")
    brand_name: str
    aliases: List[str] = Field(default_factory=list)
    category: str
    representative_products: List[str] = Field(default_factory=list)
    ambiguity_note: str
    retrieval_terms: List[str] = Field(default_factory=list)
    official_source: str
    content: str
    index_version: str


class BrandRetrievalHit(StrictModel):
    chunk: BrandKnowledgeChunk
    score: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)


def _field(body: str, name: str) -> str:
    match = re.search(rf"^- {re.escape(name)}:\s*(.+)$", body, re.MULTILINE)
    if not match:
        raise ValueError(f"brand section is missing {name}")
    return match.group(1).strip()


def load_brand_knowledge(path: Path) -> List[BrandKnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    version_match = _VERSION.search(text)
    version = version_match.group("version") if version_match else "v1.0.0"
    headings = list(_SECTION.finditer(text))
    chunks: List[BrandKnowledgeChunk] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        chunks.append(BrandKnowledgeChunk(
            chunk_id=heading.group("id"),
            brand_name=_field(body, "Canonical name"),
            aliases=[item.strip() for item in _field(body, "Aliases").split(";") if item.strip()],
            category=_field(body, "Category"),
            representative_products=[item.strip() for item in _field(body, "Representative products").split(";") if item.strip()],
            ambiguity_note=_field(body, "Ambiguity note").strip('"'),
            retrieval_terms=[item.strip() for item in _field(body, "Retrieval terms").split(",") if item.strip()],
            official_source=_field(body, "Official source"),
            content=f"{heading.group('name')}\n{body}",
            index_version=version,
        ))
    if len(chunks) != 50:
        raise ValueError(f"expected 50 public brand records, found {len(chunks)}")
    return chunks


class BrandKnowledgeStore:
    collection_name = "brand_product_chunks_v1"

    def __init__(self, persist_directory: Path, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "schema": "brand_product_v1",
                "embedding_model": self.embedder.name,
            },
        )

    def index(self, chunks: List[BrandKnowledgeChunk], replace: bool = True) -> int:
        if replace:
            try:
                self.client.delete_collection(self.collection_name)
            except ValueError:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "schema": "brand_product_v1",
                    "embedding_model": self.embedder.name,
                },
            )
        documents = [chunk.content for chunk in chunks]
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=documents,
            embeddings=self.embedder.embed_documents(documents),
            metadatas=[{"payload": chunk.model_dump_json()} for chunk in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def retrieve(self, query: str, top_k: int = 5) -> List[BrandRetrievalHit]:
        if not query.strip() or not self.count():
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            include=["metadatas", "distances"],
        )
        hits = []
        for rank, (metadata, distance) in enumerate(zip(result["metadatas"][0], result["distances"][0]), 1):
            hits.append(BrandRetrievalHit(
                chunk=BrandKnowledgeChunk.model_validate_json(metadata["payload"]),
                score=max(0.0, min(1.0, 1.0 - float(distance))),
                rank=rank,
            ))
        return hits

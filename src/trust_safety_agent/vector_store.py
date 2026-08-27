"""Chroma-backed policy index and retrieval API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import chromadb

from trust_safety_agent.embeddings import HashingEmbedder
from trust_safety_agent.schema import PolicyChunk, RetrievalHit


MetadataValue = Union[str, int, float, bool]


class PolicyVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "policy_chunks_v1",
        embedder: Optional[HashingEmbedder] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = embedder or HashingEmbedder()
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedder.name,
            },
        )

    @staticmethod
    def _to_metadata(chunk: PolicyChunk) -> Dict[str, MetadataValue]:
        return {
            "schema_version": chunk.schema_version,
            "policy_id": chunk.policy_id,
            "title": chunk.title,
            "heading_path": json.dumps(chunk.heading_path),
            "source": chunk.source,
            "policy_version": chunk.policy_version,
            "chunk_index": chunk.chunk_index,
            "policy_label": chunk.policy_label.value if chunk.policy_label else "",
            "exemption_type": chunk.exemption_type.value,
        }

    @staticmethod
    def _from_record(
        chunk_id: str,
        document: str,
        metadata: Dict[str, MetadataValue],
    ) -> PolicyChunk:
        policy_label = str(metadata["policy_label"]) or None
        return PolicyChunk(
            schema_version=str(metadata["schema_version"]),
            chunk_id=chunk_id,
            policy_id=str(metadata["policy_id"]),
            title=str(metadata["title"]),
            heading_path=json.loads(str(metadata["heading_path"])),
            source=str(metadata["source"]),
            policy_version=str(metadata["policy_version"]),
            chunk_index=int(metadata["chunk_index"]),
            content=document,
            policy_label=policy_label,
            exemption_type=str(metadata["exemption_type"]),
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self._get_or_create_collection()

    def index(self, chunks: Sequence[PolicyChunk], replace: bool = True) -> int:
        if replace:
            self.reset()
        if not chunks:
            return 0

        documents = [chunk.content for chunk in chunks]
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=documents,
            metadatas=[self._to_metadata(chunk) for chunk in chunks],
            embeddings=self.embedder.embed_documents(documents),
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def list_chunks(self) -> List[PolicyChunk]:
        if not self.count():
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        records = [
            self._from_record(chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
            )
        ]
        return sorted(records, key=lambda item: (item.policy_id, item.title, item.chunk_index))

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if not query.strip() or not self.count():
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        hits: List[RetrievalHit] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            hits.append(
                RetrievalHit(
                    chunk=self._from_record(chunk_id, document, metadata),
                    score=max(0.0, min(1.0, 1.0 - float(distance))),
                    rank=rank,
                )
            )
        return hits

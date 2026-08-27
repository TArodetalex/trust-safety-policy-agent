"""Build and inspect the persistent Policy KB."""

from __future__ import annotations

import argparse
from pathlib import Path

from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.vector_store import PolicyVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_CHROMA_DIRECTORY,
    )
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_policy_file(args.policy)
    store = PolicyVectorStore(args.persist_directory)
    indexed = store.index(chunks, replace=True)
    print(
        f"Policy KB ready: {indexed} chunks from {args.policy.name}; "
        f"embedding={store.embedder.name}; collection={store.collection_name}."
    )

    if args.query:
        for hit in store.retrieve(args.query, top_k=args.top_k):
            print(
                f"{hit.rank}. {hit.chunk.policy_id} / {hit.chunk.title} "
                f"(score={hit.score:.3f}, chunk={hit.chunk.chunk_id})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

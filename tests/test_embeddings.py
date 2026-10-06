from __future__ import annotations

from trust_safety_agent.embeddings import HashingEmbedder


def _similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def test_hashing_embedder_matches_chinese_terms() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("缺少品牌授权的商品")
    relevant = embedder.embed_query("商品未取得品牌授权，属于缺少授权风险")
    unrelated = embedder.embed_query("店铺头像无法读取，需要人工检查")

    assert embedder.name.startswith("hashing-v2-cjk")
    assert _similarity(query, relevant) > _similarity(query, unrelated)


def test_hashing_embedder_keeps_english_identifiers_searchable() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("MBA brand authorization")
    relevant = embedder.embed_query("MBA requires brand_authorized evidence")
    unrelated = embedder.embed_query("shop avatar image quality")

    assert _similarity(query, relevant) > _similarity(query, unrelated)

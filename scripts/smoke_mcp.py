"""Verify MCP handshake, discovery, resource reading, and one tool call."""

from __future__ import annotations

import asyncio
import argparse
import sys

from mcp import Client, StdioServerParameters


async def smoke(url: str | None = None) -> None:
    server = url or StdioServerParameters(
        command=sys.executable, args=["-m", "trust_safety_agent.mcp_server"]
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        names = sorted(tool.name for tool in tools.tools)
        required = {
            "classify_policy_signals",
            "get_review_queue",
            "lookup_brand",
            "review_product",
            "search_policy",
            "submit_human_review",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"missing MCP tools: {sorted(missing)}")
        result = await client.call_tool(
            "search_policy",
            {"query": "counterfeit replica product", "top_k": 3},
        )
        if result.is_error or not result.structured_content:
            raise RuntimeError("MCP search_policy returned no structured result")
        product = await client.call_tool(
            "review_product",
            {
                "case_id": "MCP-SMOKE-MBA-001",
                "title": "ASTRIA 银色头戴耳机",
                "description": "ASTRIA 品牌耳机",
                "optional_brand_field": "Astria",
                "observed_product_brand": "Astria",
                "visual_marks": ["exact Astria wordmark"],
                "brand_authorization_status": "unauthorized",
                "logo_match_type": "exact",
                "image_quality": "clear",
                "visual_confidence": 0.98,
            },
        )
        product_output = product.structured_content or {}
        if product.is_error or not (
            product_output.get("suggested_decision") == "reject"
            and product_output.get("risk_subtype")
            == "missing_brand_authorization"
            and any(
                item.get("policy_id") == "POL-MBA-001"
                for item in product_output.get("policy_references", [])
            )
        ):
            raise RuntimeError("MCP review_product did not consume Phase 8 fields")
        resources = await client.list_resources()
        resource_uris = sorted(str(item.uri) for item in resources.resources)
        transport = "HTTP" if url else "stdio"
        print(f"MCP {transport} handshake ok; tools={len(names)}; resources={len(resource_uris)}")
        print("search_policy structured output ok")
        print("review_product Phase 8 structured fields ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    args = parser.parse_args()
    asyncio.run(smoke(args.url))

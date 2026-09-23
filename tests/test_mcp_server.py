import asyncio
from pathlib import Path

from mcp import Client

from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import DEFAULT_BRAND_LIBRARY_PATH, DEFAULT_POLICY_PATH
from trust_safety_agent.mcp_server import create_mcp_server
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.reviewer_workspace import ReviewerWorkspaceStore
from trust_safety_agent.vector_store import PolicyVectorStore


def test_mcp_server_discovers_and_calls_structured_tool(tmp_path: Path):
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(DEFAULT_POLICY_PATH))
    server = create_mcp_server(
        store,
        ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH),
        ReviewerWorkspaceStore(tmp_path / "reviews.jsonl"),
    )

    async def exercise():
        async with Client(server) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert "search_policy" in names
            assert "classify_policy_signals" in names
            assert "submit_human_review" in names
            result = await client.call_tool(
                "search_policy",
                {"query": "counterfeit replica", "top_k": 3},
            )
            assert not result.is_error
            assert result.structured_content["hits"]
            signal = await client.call_tool(
                "classify_policy_signals",
                {
                    "input_text": "Gucci dupe same design as the original",
                    "content_type": "product",
                },
            )
            assert not signal.is_error
            assert signal.structured_content["policy_label"] == "knockoff"
            resources = await client.list_resources()
            assert {str(item.uri) for item in resources.resources} == {
                "policy://catalog",
                "review://pending",
            }

    asyncio.run(exercise())

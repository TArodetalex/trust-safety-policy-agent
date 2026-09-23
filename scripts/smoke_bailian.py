"""Validate Bailian configuration and optionally run one multimodal API call."""

from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import asdict
from pathlib import Path

from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.llm_adjudicator import LLMPolicyAdjudicator
from trust_safety_agent.llm_client import ImageInput
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.resilient_llm_client import (
    ResilientLLMSettings,
    ResilientOpenAICompatibleClient,
)
from trust_safety_agent.schema import ContentType
from trust_safety_agent.vector_store import PolicyVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "data/golden_set/assets/v4/IMG-C151.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send exactly one adjudication request to the configured provider.",
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--text",
        default="Product submitted for a controlled visual counterfeit review.",
    )
    return parser.parse_args()


def safe_settings(settings: ResilientLLMSettings) -> dict[str, object]:
    return {
        "api_key_configured": bool(settings.api_key.strip()),
        "api_base": settings.api_base,
        "primary_model": settings.model,
        "fallback_models": list(settings.fallback_models),
        "response_format": settings.response_format,
        "timeout_seconds": settings.timeout_seconds,
    }


def main() -> int:
    args = parse_args()
    settings = ResilientLLMSettings.from_env()
    settings.validate()
    print(json.dumps(safe_settings(settings), indent=2, ensure_ascii=False))
    if not args.live:
        if settings.api_key.strip():
            print("Configuration is valid. Add --live to send one billable request.")
        else:
            print("Configuration file is valid; add LLM_API_KEY before a live call.")
        return 0
    if not settings.api_key.strip():
        raise SystemExit("LLM_API_KEY is missing from the local .env file.")
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"Smoke-test image does not exist: {image_path}")

    store = PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)
    if store.count() == 0:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    client = ResilientOpenAICompatibleClient(settings)
    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C151",
        content_type=ContentType.PRODUCT,
        input_text=args.text,
        images=[ImageInput(data=image_path.read_bytes(), media_type=media_type)],
    )
    print(decision.model_dump_json(indent=2))
    print(json.dumps(asdict(client.last_metadata), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

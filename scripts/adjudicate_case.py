"""Run one policy-grounded case decision from the command line."""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.llm_adjudicator import LLMPolicyAdjudicator
from trust_safety_agent.llm_client import (
    ImageInput,
    LLMSettings,
    OpenAICompatibleClient,
)
from trust_safety_agent.ocr_adjudicator import (
    MacOSVisionOCR,
    OCRPolicyAdjudicator,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.production_router import (
    ProductionRouter,
    ProductionRoutingConfig,
)
from trust_safety_agent.schema import ContentType
from trust_safety_agent.vector_store import PolicyVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTING_CONFIG = PROJECT_ROOT / "config/production_routing_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="", help="Content to assess")
    parser.add_argument(
        "--content-type",
        choices=[item.value for item in ContentType],
        default=ContentType.PRODUCT.value,
    )
    parser.add_argument("--case-id", default="C000")
    parser.add_argument(
        "--engine",
        choices=["rules", "production", "llm"],
        default="rules",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--image", type=Path, help="Local image path")
    image_group.add_argument("--image-url", help="Remote image URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)
    if store.count() == 0:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))
    if not args.text.strip() and not args.image and not args.image_url:
        raise SystemExit("Provide text, --image, or --image-url.")

    if args.engine == "production":
        settings = LLMSettings.from_env()
        llm_agent = (
            LLMPolicyAdjudicator(
                store,
                OpenAICompatibleClient(settings),
            )
            if settings.configured
            else None
        )
        rules = PolicyAdjudicator(store)
        result = ProductionRouter(
            config=ProductionRoutingConfig.from_json(ROUTING_CONFIG),
            rules=rules,
            ocr=OCRPolicyAdjudicator(
                rules,
                MacOSVisionOCR(
                    PROJECT_ROOT / "scripts/macos_vision_ocr.swift"
                ),
            ),
            llm=llm_agent,
        ).adjudicate(
            case_id=args.case_id,
            content_type=ContentType(args.content_type),
            input_text=args.text,
            image_path=args.image,
            image_url=args.image_url,
        )
        print(result.model_dump_json(indent=2))
        return 0

    images = []
    if args.image:
        media_type = mimetypes.guess_type(args.image.name)[0] or "image/jpeg"
        images.append(ImageInput(data=args.image.read_bytes(), media_type=media_type))
    elif args.image_url:
        images.append(ImageInput(url=args.image_url))

    if args.engine == "llm":
        settings = LLMSettings.from_env()
        if not settings.configured:
            raise SystemExit(
                "Set LLM_API_KEY and LLM_MODEL before using --engine llm."
            )
        decision = LLMPolicyAdjudicator(
            store,
            OpenAICompatibleClient(settings),
        ).adjudicate(
            case_id=args.case_id,
            content_type=ContentType(args.content_type),
            input_text=args.text,
            images=images,
        )
    else:
        if images:
            raise SystemExit("Image inputs require --engine llm.")
        decision = PolicyAdjudicator(store).adjudicate(
            case_id=args.case_id,
            content_type=ContentType(args.content_type),
            input_text=args.text,
        )
    print(decision.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backward-compatible public entry point for multimodal LLM clients."""

from trust_safety_agent.llm_core import (
    ImageInput,
    LLMClientError,
    MultimodalChatClient,
)
from trust_safety_agent.resilient_llm_client import (
    ResilientLLMCallMetadata as LLMCallMetadata,
    ResilientLLMSettings as LLMSettings,
    ResilientOpenAICompatibleClient as OpenAICompatibleClient,
)

__all__ = [
    "ImageInput",
    "LLMCallMetadata",
    "LLMClientError",
    "LLMSettings",
    "MultimodalChatClient",
    "OpenAICompatibleClient",
]

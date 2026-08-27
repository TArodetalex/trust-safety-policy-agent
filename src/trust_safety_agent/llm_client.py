"""OpenAI-compatible multimodal chat client."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib.parse import urlparse


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class LLMClientError(RuntimeError):
    """A provider error safe to surface without credentials."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = DEFAULT_MODEL
    timeout_seconds: int = 60
    require_parameters: bool = True
    allow_fallbacks: bool = False
    provider_order: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            api_base=(
                os.getenv("LLM_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or DEFAULT_API_BASE
            ),
            model=(
                os.getenv("VISION_MODEL")
                or os.getenv("LLM_MODEL")
                or os.getenv("OPENAI_MODEL")
                or DEFAULT_MODEL
            ),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            require_parameters=(
                os.getenv("LLM_REQUIRE_PARAMETERS", "true").casefold() == "true"
            ),
            allow_fallbacks=(
                os.getenv("LLM_ALLOW_FALLBACKS", "false").casefold() == "true"
            ),
            provider_order=tuple(
                provider.strip()
                for provider in os.getenv("LLM_PROVIDER_ORDER", "").split(",")
                if provider.strip()
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip() and self.model.strip())

    def validate(self) -> None:
        parsed = urlparse(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM API base must be an http(s) URL")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("LLM timeout must be between 1 and 300 seconds")


@dataclass(frozen=True)
class ImageInput:
    url: Optional[str] = None
    data: Optional[bytes] = None
    media_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        if bool(self.url) == bool(self.data):
            raise ValueError("image input requires exactly one of url or data")
        if self.data and len(self.data) > MAX_IMAGE_BYTES:
            raise ValueError("uploaded image exceeds the 8 MB limit")
        if not self.media_type.startswith("image/"):
            raise ValueError("uploaded file must use an image media type")
        if self.url:
            parsed = urlparse(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("image URL must be an http(s) URL")

    def as_message_part(self) -> Dict[str, object]:
        if self.url:
            image_url = self.url
        else:
            encoded = base64.b64encode(self.data or b"").decode("ascii")
            image_url = f"data:{self.media_type};base64,{encoded}"
        return {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }


@dataclass(frozen=True)
class LLMCallMetadata:
    requested_model: str
    response_model: Optional[str] = None
    provider: Optional[str] = None
    request_id: Optional[str] = None
    status_code: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None


class MultimodalChatClient(Protocol):
    model: str
    last_metadata: LLMCallMetadata

    def complete_json(
        self,
        system_prompt: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "trust_safety_decision",
    ) -> Dict[str, Any]:
        ...


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings) -> None:
        settings.validate()
        if not settings.configured:
            raise ValueError("LLM API key and model are required")
        self.settings = settings
        self.model = settings.model
        self.last_metadata = LLMCallMetadata(requested_model=self.model)

    @property
    def endpoint(self) -> str:
        return f"{self.settings.api_base.rstrip('/')}/chat/completions"

    @staticmethod
    def _extract_json(content: object) -> Dict[str, Any]:
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        if not isinstance(content, str):
            raise LLMClientError("LLM response did not contain text content")
        cleaned = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise LLMClientError("LLM response JSON must be an object")
        return payload

    def complete_json(
        self,
        system_prompt: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "trust_safety_decision",
    ) -> Dict[str, Any]:
        self.last_metadata = LLMCallMetadata(requested_model=self.model)
        user_content: List[Dict[str, object]] = [
            {"type": "text", "text": user_text}
        ]
        user_content.extend(image.as_message_part() for image in images)
        response_format: Dict[str, object]
        if response_schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        body: Dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 900,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if urlparse(self.settings.api_base).netloc.endswith("openrouter.ai"):
            provider: Dict[str, object] = {
                "require_parameters": self.settings.require_parameters,
                "allow_fallbacks": self.settings.allow_fallbacks,
            }
            if self.settings.provider_order:
                provider["order"] = list(self.settings.provider_order)
            body["provider"] = provider
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        status_code: Optional[int] = None
        request_id: Optional[str] = None
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                status_code = getattr(response, "status", 200)
                headers = getattr(response, "headers", None)
                if headers is not None:
                    request_id = headers.get("x-request-id")
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self.last_metadata = LLMCallMetadata(
                requested_model=self.model,
                status_code=exc.code,
            )
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMClientError(
                f"LLM API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(f"LLM API connection failed: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError("LLM API returned an unreadable response") from exc

        usage = response_body.get("usage") or {}
        self.last_metadata = LLMCallMetadata(
            requested_model=self.model,
            response_model=response_body.get("model"),
            provider=response_body.get("provider"),
            request_id=request_id,
            status_code=status_code,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cost=(
                float(usage["cost"])
                if usage.get("cost") is not None
                else None
            ),
        )
        try:
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM API response is missing message content") from exc
        return self._extract_json(content)

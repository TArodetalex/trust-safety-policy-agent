"""Provider-aware OpenAI-compatible client with controlled model fallback."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv

from trust_safety_agent.llm_core import (
    ImageInput,
    LLMCallMetadata,
    LLMClientError,
    LLMSettings,
    OpenAICompatibleClient,
)


RESPONSE_FORMAT_MODES = {"auto", "json_schema", "json_object"}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)\bsk-[a-z0-9._-]{8,}\b"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,}\"']+"),
)


@dataclass(frozen=True)
class ResilientLLMSettings(LLMSettings):
    response_format: str = "auto"
    fallback_models: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ResilientLLMSettings":
        load_dotenv(override=False)
        base = LLMSettings.from_env()
        return cls(
            api_key=base.api_key,
            api_base=base.api_base,
            model=base.model,
            timeout_seconds=base.timeout_seconds,
            require_parameters=base.require_parameters,
            allow_fallbacks=base.allow_fallbacks,
            provider_order=base.provider_order,
            response_format=os.getenv("LLM_RESPONSE_FORMAT", "auto").casefold(),
            fallback_models=tuple(
                model.strip()
                for model in os.getenv("LLM_FALLBACK_MODELS", "").split(",")
                if model.strip()
            ),
        )

    def validate(self) -> None:
        super().validate()
        if self.response_format not in RESPONSE_FORMAT_MODES:
            raise ValueError(
                "LLM response format must be auto, json_schema, or json_object"
            )
        names = (self.model, *self.fallback_models)
        if len(set(names)) != len(names):
            raise ValueError("LLM primary and fallback model names must be unique")


@dataclass(frozen=True)
class ResilientLLMCallMetadata(LLMCallMetadata):
    selected_model: Optional[str] = None
    response_format: Optional[str] = None
    attempted_models: tuple[str, ...] = ()
    attempt_count: int = 0
    fallback_reason: Optional[str] = None


class ResilientOpenAICompatibleClient(OpenAICompatibleClient):
    """Use an explicit model allowlist and never discover models from a key."""

    def __init__(self, settings: ResilientLLMSettings) -> None:
        super().__init__(settings)
        self.settings = settings
        self.last_metadata = ResilientLLMCallMetadata(
            requested_model=self.model
        )

    @property
    def provider_name(self) -> str:
        host = urlparse(self.settings.api_base).netloc.casefold()
        if "aliyuncs.com" in host or "dashscope" in host:
            return "Alibaba Cloud Model Studio"
        if host.endswith("openrouter.ai"):
            return "OpenRouter"
        return host or "unknown"

    def _redact(self, value: str) -> str:
        cleaned = value.replace(self.settings.api_key, "[REDACTED]")
        for pattern in SECRET_PATTERNS:
            cleaned = pattern.sub(
                lambda match: (
                    f"{match.group(1)}[REDACTED]"
                    if match.lastindex
                    else "[REDACTED]"
                ),
                cleaned,
            )
        return cleaned[:400]

    def _safe_error_detail(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self._redact(raw)
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if not isinstance(error, dict):
            return "provider returned an error"
        code = self._redact(str(error.get("code") or "provider_error"))
        message = self._redact(str(error.get("message") or "request failed"))
        return f"{code}: {message}"

    @staticmethod
    def _quota_error(status_code: int, detail: str) -> bool:
        lowered = detail.casefold()
        return status_code == 403 and any(
            marker in lowered
            for marker in ("allocationquota", "free tier", "quota", "allocated")
        )

    def _format_modes(
        self,
        response_schema: Optional[Dict[str, Any]],
    ) -> tuple[str, ...]:
        if response_schema is None:
            return ("json_object",)
        if self.settings.response_format != "auto":
            return (self.settings.response_format,)
        host = urlparse(self.settings.api_base).netloc.casefold()
        if "aliyuncs.com" in host or "dashscope" in host:
            return ("json_object",)
        return ("json_schema", "json_object")

    @staticmethod
    def _response_format(
        mode: str,
        response_schema: Optional[Dict[str, Any]],
        schema_name: str,
    ) -> Dict[str, object]:
        if mode == "json_schema" and response_schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        return {"type": "json_object"}

    @staticmethod
    def _format_error(status_code: int, detail: str) -> bool:
        lowered = detail.casefold()
        return status_code == 400 and any(
            marker in lowered
            for marker in ("response_format", "json_schema", "json mode")
        )

    def _failure_metadata(
        self,
        model: str,
        status_code: Optional[int],
        format_mode: str,
        attempted_models: List[str],
        attempt_count: int,
        reason: str,
    ) -> None:
        self.last_metadata = ResilientLLMCallMetadata(
            requested_model=self.model,
            selected_model=model,
            provider=self.provider_name,
            status_code=status_code,
            response_format=format_mode,
            attempted_models=tuple(attempted_models),
            attempt_count=attempt_count,
            fallback_reason=reason,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "trust_safety_decision",
    ) -> Dict[str, Any]:
        user_content: List[Dict[str, object]] = [
            {"type": "text", "text": user_text}
        ]
        user_content.extend(image.as_message_part() for image in images)
        models = (self.model, *self.settings.fallback_models)
        formats = self._format_modes(response_schema)
        attempted_models: List[str] = []
        attempt_count = 0
        fallback_reason: Optional[str] = None
        last_error: Optional[LLMClientError] = None

        for model_index, model in enumerate(models):
            attempted_models.append(model)
            for format_index, format_mode in enumerate(formats):
                attempt_count += 1
                body: Dict[str, object] = {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 900,
                    "response_format": self._response_format(
                        format_mode, response_schema, schema_name
                    ),
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
                    content = response_body["choices"][0]["message"]["content"]
                    payload = self._extract_json(content)
                except urllib.error.HTTPError as exc:
                    detail = self._safe_error_detail(
                        exc.read().decode("utf-8", errors="replace")
                    )
                    fallback_reason = f"HTTP {exc.code}: {detail}"
                    last_error = LLMClientError(
                        f"LLM API returned {fallback_reason}"
                    )
                    if self._format_error(exc.code, detail) and format_index + 1 < len(formats):
                        continue
                    retry_model = (
                        exc.code in RETRYABLE_STATUS_CODES
                        or self._quota_error(exc.code, detail)
                    )
                    if retry_model and model_index + 1 < len(models):
                        break
                    self._failure_metadata(
                        model, exc.code, format_mode, attempted_models,
                        attempt_count, fallback_reason,
                    )
                    raise last_error from exc
                except urllib.error.URLError as exc:
                    fallback_reason = f"connection failed: {self._redact(str(exc.reason))}"
                    last_error = LLMClientError(f"LLM API {fallback_reason}")
                    if model_index + 1 < len(models):
                        break
                    self._failure_metadata(
                        model, None, format_mode, attempted_models,
                        attempt_count, fallback_reason,
                    )
                    raise last_error from exc
                except (TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    fallback_reason = "provider returned an unreadable response"
                    last_error = LLMClientError(f"LLM API {fallback_reason}")
                    if model_index + 1 < len(models):
                        break
                    self._failure_metadata(
                        model, status_code, format_mode, attempted_models,
                        attempt_count, fallback_reason,
                    )
                    raise last_error from exc
                except LLMClientError as exc:
                    fallback_reason = self._redact(str(exc))
                    last_error = exc
                    if model_index + 1 < len(models):
                        break
                    self._failure_metadata(
                        model, status_code, format_mode, attempted_models,
                        attempt_count, fallback_reason,
                    )
                    raise
                else:
                    usage = response_body.get("usage") or {}
                    self.last_metadata = ResilientLLMCallMetadata(
                        requested_model=self.model,
                        selected_model=model,
                        response_model=response_body.get("model"),
                        provider=response_body.get("provider") or self.provider_name,
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
                        response_format=format_mode,
                        attempted_models=tuple(attempted_models),
                        attempt_count=attempt_count,
                        fallback_reason=(
                            fallback_reason if model != self.model else None
                        ),
                    )
                    return payload

        raise last_error or LLMClientError("LLM API request failed")

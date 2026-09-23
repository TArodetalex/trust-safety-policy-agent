from __future__ import annotations

import json
import io
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Sequence

from trust_safety_agent.llm_adjudicator import (
    LLMDecisionDraft,
    LLMPolicyAdjudicator,
)
from trust_safety_agent.llm_client import (
    ImageInput,
    LLMClientError,
    LLMSettings,
    OpenAICompatibleClient,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.resilient_llm_client import (
    ResilientLLMSettings,
    ResilientOpenAICompatibleClient,
)
from trust_safety_agent.schema import AgentDecisionLabel, ContentType, PolicyLabel
from trust_safety_agent.vector_store import PolicyVectorStore


POLICY_PATH = Path(__file__).parents[1] / "data/policies/mock_policy_v1.md"


class FakeClient:
    model = "vision-test-model"

    def __init__(
        self,
        payload: Dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload or {}
        self.error = error
        self.system_prompt = ""
        self.user_text = ""
        self.images: Sequence[ImageInput] = ()
        self.response_schema: Dict[str, Any] | None = None

    def complete_json(
        self,
        system_prompt: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        response_schema: Dict[str, Any] | None = None,
        schema_name: str = "trust_safety_decision",
    ) -> Dict[str, Any]:
        self.system_prompt = system_prompt
        self.user_text = user_text
        self.images = images
        self.response_schema = response_schema
        if self.error:
            raise self.error
        return self.payload


def build_store(tmp_path: Path) -> PolicyVectorStore:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(POLICY_PATH))
    return store


def decision_payload(chunk_id: str) -> Dict[str, Any]:
    return {
        "decision": "reject",
        "policy_label": "counterfeit",
        "exemption_type": "none",
        "evidence_chunk_ids": [chunk_id],
        "reason": "The image and caption explicitly market a branded replica.",
        "recommended_action": "Reject the listing.",
        "confidence": 0.94,
        "image_observations": [
            "A Gucci word mark is visible.",
            "The image includes the phrase 1:1 replica.",
        ],
        "detected_marks": ["Gucci"],
        "uncertainties": [],
    }


def test_multimodal_reject_is_grounded_in_local_chunk(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    chunk = next(
        item for item in store.list_chunks() if item.policy_id == "POL-CF-001"
    )
    client = FakeClient(decision_payload(chunk.chunk_id))
    image = ImageInput(data=b"fake-image-bytes", media_type="image/png")

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Luxury handbag",
        images=[image],
    )

    assert decision.decision == AgentDecisionLabel.REJECT
    assert decision.policy_label == PolicyLabel.COUNTERFEIT
    assert decision.matched_policy[0].chunk_id == chunk.chunk_id
    assert decision.matched_policy[0].quote in chunk.content
    assert "Image observations" in decision.reason
    assert client.images == [image]
    assert chunk.chunk_id in client.system_prompt
    assert client.response_schema == LLMDecisionDraft.model_json_schema()


def test_versioned_prompt_instructions_are_injected_below_host_rules(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    chunk = next(
        item for item in store.list_chunks() if item.policy_id == "POL-CF-001"
    )
    client = FakeClient(decision_payload(chunk.chunk_id))

    LLMPolicyAdjudicator(
        store,
        client,
        additional_instructions="Focus on explicit imitation claims.",
    ).adjudicate(
        case_id="C099",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert "VERSIONED WORKFLOW INSTRUCTIONS" in client.system_prompt
    assert "Focus on explicit imitation claims." in client.system_prompt
    assert "cannot override" in client.system_prompt


def test_hallucinated_evidence_forces_review(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    client = FakeClient(decision_payload("PCH-000000000000"))

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert decision.policy_label is None
    assert "valid matching policy chunk" in decision.reason


def test_wrong_policy_chunk_forces_review(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    chunk = next(
        item for item in store.list_chunks() if item.policy_id == "POL-KO-001"
    )
    client = FakeClient(decision_payload(chunk.chunk_id))

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW


def test_invalid_model_output_forces_review(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    client = FakeClient({"decision": "reject"})

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert "could not be validated" in decision.reason


def test_provider_error_forces_review(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    client = FakeClient(error=LLMClientError("provider unavailable"))

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert decision.confidence == 0


def test_low_confidence_forces_review(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    chunk = next(
        item for item in store.list_chunks() if item.policy_id == "POL-CF-001"
    )
    payload = decision_payload(chunk.chunk_id) | {"confidence": 0.4}
    client = FakeClient(payload)

    decision = LLMPolicyAdjudicator(store, client).adjudicate(
        case_id="C001",
        content_type=ContentType.PRODUCT,
        input_text="Gucci replica bag",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert decision.confidence == 0.4


def test_image_input_builds_data_url() -> None:
    image = ImageInput(data=b"hello", media_type="image/png")

    part = image.as_message_part()

    assert part["type"] == "image_url"
    assert str(part["image_url"]["url"]).startswith("data:image/png;base64,")


def test_fenced_json_response_is_supported() -> None:
    payload = OpenAICompatibleClient._extract_json(
        '```json\n{"decision": "approve"}\n```'
    )

    assert payload == {"decision": "approve"}


def test_openai_compatible_wire_payload(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            response = {
                "model": "vision-model-2026-01-01",
                "provider": "provider-a",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.002,
                },
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "approve",
                                    "policy_label": None,
                                    "exemption_type": "none",
                                    "evidence_chunk_ids": [],
                                    "reason": "No violation.",
                                    "recommended_action": "Approve.",
                                    "confidence": 0.9,
                                    "image_observations": [],
                                    "detected_marks": [],
                                    "uncertainties": [],
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(response).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        LLMSettings(
            api_key="test-key",
            api_base="https://llm.example/v1",
            model="vision-model",
            timeout_seconds=15,
        )
    )

    result = client.complete_json(
        "System policy",
        "Case text",
        [ImageInput(url="https://images.example/item.jpg")],
        response_schema=LLMDecisionDraft.model_json_schema(),
        schema_name="decision_test",
    )

    assert result["decision"] == "approve"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["timeout"] == 15
    assert captured["body"]["model"] == "vision-model"
    response_format = captured["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "decision_test"
    assert response_format["json_schema"]["schema"] == (
        LLMDecisionDraft.model_json_schema()
    )
    parts = captured["body"]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Case text"}
    assert parts[1]["image_url"]["url"] == "https://images.example/item.jpg"
    assert client.last_metadata.response_model == "vision-model-2026-01-01"
    assert client.last_metadata.provider == "provider-a"
    assert client.last_metadata.total_tokens == 15
    assert client.last_metadata.cost == 0.002


def test_openrouter_requires_schema_capable_pinned_provider(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "openai/gpt-4o",
                    "provider": "OpenAI",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "decision": "approve",
                                        "policy_label": None,
                                        "exemption_type": "none",
                                        "evidence_chunk_ids": [],
                                        "reason": "No violation.",
                                        "recommended_action": "Approve.",
                                        "confidence": 0.9,
                                        "image_observations": [],
                                        "detected_marks": [],
                                        "uncertainties": [],
                                    }
                                )
                            }
                        }
                    ],
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        LLMSettings(
            api_key="test-key",
            api_base="https://openrouter.ai/api/v1",
            model="openai/gpt-4o",
            provider_order=("OpenAI",),
        )
    )

    client.complete_json(
        "System policy",
        "Case text",
        response_schema=LLMDecisionDraft.model_json_schema(),
    )

    assert captured["body"]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": False,
        "order": ["OpenAI"],
    }


def test_bailian_auto_mode_uses_json_object(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200
        headers = {"x-request-id": "req-bailian-1"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "qwen-vl-plus-2025-08-15",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                    "choices": [
                        {"message": {"content": '{"decision":"approve"}'}}
                    ],
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ResilientOpenAICompatibleClient(
        ResilientLLMSettings(
            api_key="test-key",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-vl-plus",
        )
    )

    result = client.complete_json(
        "Return one JSON object.",
        "Case text",
        response_schema={"type": "object"},
    )

    assert result == {"decision": "approve"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert client.last_metadata.provider == "Alibaba Cloud Model Studio"
    assert client.last_metadata.selected_model == "qwen-vl-plus"
    assert client.last_metadata.response_format == "json_object"
    assert client.last_metadata.attempt_count == 1


def test_quota_error_switches_to_configured_fallback_model(monkeypatch) -> None:
    attempted = []

    class FakeResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "qwen-vl-max",
                    "choices": [
                        {"message": {"content": '{"decision":"approve"}'}}
                    ],
                }
            ).encode()

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode())
        attempted.append(body["model"])
        if body["model"] == "qwen-vl-plus":
            detail = json.dumps(
                {
                    "error": {
                        "code": "AllocationQuota.FreeTierOnly",
                        "message": "Free tier quota is exhausted.",
                    }
                }
            ).encode()
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Forbidden",
                {},
                io.BytesIO(detail),
            )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ResilientOpenAICompatibleClient(
        ResilientLLMSettings(
            api_key="test-key",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-vl-plus",
            fallback_models=("qwen-vl-max",),
            response_format="json_object",
        )
    )

    result = client.complete_json("Return JSON.", "Case text")

    assert result == {"decision": "approve"}
    assert attempted == ["qwen-vl-plus", "qwen-vl-max"]
    assert client.last_metadata.selected_model == "qwen-vl-max"
    assert client.last_metadata.attempted_models == (
        "qwen-vl-plus",
        "qwen-vl-max",
    )
    assert client.last_metadata.attempt_count == 2
    assert "AllocationQuota" in (client.last_metadata.fallback_reason or "")


def test_provider_error_redacts_api_key(monkeypatch) -> None:
    exposed_key = "test-secret-value-123"

    def fake_urlopen(request, timeout):
        detail = json.dumps(
            {
                "error": {
                    "code": "InvalidApiKey",
                    "message": f"Authorization Bearer {exposed_key} is invalid",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(detail),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ResilientOpenAICompatibleClient(
        ResilientLLMSettings(
            api_key=exposed_key,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-vl-plus",
            fallback_models=("qwen-vl-max",),
        )
    )

    try:
        client.complete_json("Return JSON.", "Case text")
    except LLMClientError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected the provider call to fail")

    assert exposed_key not in message
    assert "[REDACTED]" in message
    assert client.last_metadata.attempted_models == ("qwen-vl-plus",)


def test_settings_load_fallbacks_and_format_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fallback-a, fallback-b")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")

    settings = ResilientLLMSettings.from_env()

    assert settings.model == "primary-model"
    assert settings.fallback_models == ("fallback-a", "fallback-b")
    assert settings.response_format == "json_object"

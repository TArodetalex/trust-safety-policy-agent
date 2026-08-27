from __future__ import annotations

from pathlib import Path
from typing import Sequence

from trust_safety_agent.llm_client import ImageInput
from trust_safety_agent.production_router import (
    ProductionRouter,
    ProductionRoutingConfig,
)
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    ExemptionType,
    MatchedPolicyEvidence,
    PolicyLabel,
)


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config/production_routing_v1.json"


def evidence(
    exemption_type: ExemptionType = ExemptionType.NONE,
) -> list[MatchedPolicyEvidence]:
    return [
        MatchedPolicyEvidence(
            policy_id="POL-CF-001",
            chunk_id="PCH-000000000001",
            quote="Explicit counterfeit promotion is prohibited.",
            exemption_type=exemption_type,
        )
    ]


def decision(
    label: AgentDecisionLabel,
    confidence: float,
    *,
    with_evidence: bool = False,
) -> AgentDecision:
    is_reject = label == AgentDecisionLabel.REJECT
    return AgentDecision(
        case_id="C001",
        decision=label,
        policy_label=PolicyLabel.COUNTERFEIT if is_reject else None,
        matched_policy=(
            evidence(ExemptionType.COMPATIBILITY)
            if with_evidence and not is_reject
            else evidence() if is_reject else []
        ),
        reason=f"Fake {label.value} decision.",
        recommended_action="Apply the fake decision.",
        confidence=confidence,
    )


class FakeRules:
    def __init__(self, result: AgentDecision) -> None:
        self.result = result
        self.calls = 0

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
    ) -> AgentDecision:
        self.calls += 1
        return self.result


class FakeOCR:
    def __init__(self, result: AgentDecision) -> None:
        self.result = result
        self.calls = 0

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        image_path: Path,
    ) -> AgentDecision:
        self.calls += 1
        return self.result


class FakeLLM:
    def __init__(self, result: AgentDecision) -> None:
        self.result = result
        self.calls = 0
        self.images: Sequence[ImageInput] = ()

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> AgentDecision:
        self.calls += 1
        self.images = images
        return self.result


def router(
    rules: FakeRules,
    ocr: FakeOCR | None = None,
    llm: FakeLLM | None = None,
) -> ProductionRouter:
    return ProductionRouter(
        ProductionRoutingConfig.from_json(CONFIG_PATH),
        rules=rules,
        ocr=ocr,
        llm=llm,
    )


def test_text_uses_rules_without_escalating() -> None:
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    llm = FakeLLM(decision(AgentDecisionLabel.REJECT, 0.99))

    result = router(rules, llm=llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Plain canvas bag",
    )

    assert result.selected_engine == "rules"
    assert result.decision.decision == AgentDecisionLabel.APPROVE
    assert not result.escalated
    assert rules.calls == 1
    assert llm.calls == 0


def test_ambiguous_text_routes_directly_to_human_review() -> None:
    rules = FakeRules(decision(AgentDecisionLabel.NEED_REVIEW, 0.42))
    llm = FakeLLM(decision(AgentDecisionLabel.APPROVE, 0.99))

    result = router(rules, llm=llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Branded item with unclear authorization",
    )

    assert result.selected_engine == "human_review"
    assert result.decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert llm.calls == 0


def test_local_image_accepts_grounded_ocr_without_llm(tmp_path: Path) -> None:
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    ocr = FakeOCR(decision(AgentDecisionLabel.REJECT, 0.98))
    llm = FakeLLM(decision(AgentDecisionLabel.APPROVE, 0.99))

    result = router(rules, ocr, llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Visual listing",
        image_path=tmp_path / "unused.png",
    )

    assert result.selected_engine == "rules-ocr"
    assert result.decision.decision == AgentDecisionLabel.REJECT
    assert ocr.calls == 1
    assert llm.calls == 0


def test_uncertain_ocr_escalates_to_grounded_llm(tmp_path: Path) -> None:
    image_path = tmp_path / "item.png"
    image_path.write_bytes(b"image")
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    ocr = FakeOCR(decision(AgentDecisionLabel.NEED_REVIEW, 0.2))
    llm = FakeLLM(decision(AgentDecisionLabel.REJECT, 0.93))

    result = router(rules, ocr, llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Visual listing",
        image_path=image_path,
    )

    assert result.selected_engine == "multimodal-llm"
    assert result.decision.decision == AgentDecisionLabel.REJECT
    assert result.escalated
    assert [step.engine for step in result.steps] == [
        "rules-ocr",
        "multimodal-llm",
    ]
    assert llm.images[0].data == b"image"


def test_unsubstantiated_image_approval_fails_closed(tmp_path: Path) -> None:
    image_path = tmp_path / "item.png"
    image_path.write_bytes(b"image")
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    ocr = FakeOCR(decision(AgentDecisionLabel.NEED_REVIEW, 0.2))
    llm = FakeLLM(decision(AgentDecisionLabel.APPROVE, 0.99))

    result = router(rules, ocr, llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Visual listing",
        image_path=image_path,
    )

    assert result.selected_engine == "human_review"
    assert result.decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert "No configured engine" in result.decision.reason
    assert result.steps[-1].engine == "human_review"


def test_specific_image_exemption_can_be_auto_approved(
    tmp_path: Path,
) -> None:
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    ocr = FakeOCR(
        decision(
            AgentDecisionLabel.APPROVE,
            0.94,
            with_evidence=True,
        )
    )

    result = router(rules, ocr).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Compatible accessory",
        image_path=tmp_path / "unused.png",
    )

    assert result.selected_engine == "rules-ocr"
    assert result.decision.decision == AgentDecisionLabel.APPROVE


def test_llm_exemption_must_be_reproduced_by_rules(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "item.png"
    image_path.write_bytes(b"image")
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))
    ocr = FakeOCR(decision(AgentDecisionLabel.NEED_REVIEW, 0.2))
    llm = FakeLLM(
        decision(
            AgentDecisionLabel.APPROVE,
            0.96,
            with_evidence=True,
        )
    )

    result = router(rules, ocr, llm).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Unbranded ceramic item",
        image_path=image_path,
    )

    assert result.selected_engine == "human_review"
    assert result.steps[-2].engine == "visual-evidence-guard"
    assert not result.steps[-2].accepted


def test_remote_image_without_llm_routes_to_review() -> None:
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))

    result = router(rules).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Remote visual listing",
        image_url="https://images.example/item.png",
    )

    assert result.selected_engine == "human_review"
    assert result.decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert result.steps[-1].detail == "fail-closed terminal route"


def test_two_image_sources_are_rejected_by_route_guard(tmp_path: Path) -> None:
    rules = FakeRules(decision(AgentDecisionLabel.APPROVE, 0.86))

    result = router(rules).adjudicate(
        "C001",
        ContentType.PRODUCT,
        "Invalid case",
        image_path=tmp_path / "item.png",
        image_url="https://images.example/item.png",
    )

    assert result.selected_engine == "human_review"
    assert result.steps[0].engine == "route_guard"
    assert rules.calls == 0

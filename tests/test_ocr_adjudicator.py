from __future__ import annotations

from pathlib import Path

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.ocr_adjudicator import (
    OCRPolicyAdjudicator,
    OCRResult,
    OCRUnavailableError,
)
from trust_safety_agent.policy_loader import load_policy_file
from trust_safety_agent.schema import (
    AgentDecisionLabel,
    ContentType,
    PolicyLabel,
)
from trust_safety_agent.vector_store import PolicyVectorStore


POLICY_PATH = Path(__file__).parents[1] / "data/policies/mock_policy_v1.md"


class FakeExtractor:
    def __init__(
        self,
        text: str = "",
        confidence: float = 0.99,
        error: bool = False,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.error = error

    def extract(self, image_path: Path) -> OCRResult:
        if self.error:
            raise OCRUnavailableError("test failure")
        return OCRResult(
            text=self.text,
            confidence=self.confidence,
            line_count=1,
        )


def build_agent(tmp_path: Path, extractor: FakeExtractor) -> OCRPolicyAdjudicator:
    store = PolicyVectorStore(tmp_path / "chroma")
    store.index(load_policy_file(POLICY_PATH))
    return OCRPolicyAdjudicator(PolicyAdjudicator(store), extractor)


def test_explicit_ocr_violation_is_auto_rejected(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        FakeExtractor("GUCCI 1:1 MIRROR COPY branded dust bag"),
    )

    decision = agent.adjudicate(
        "C072",
        ContentType.PRODUCT,
        "Product submitted for visual policy review.",
        tmp_path / "image.png",
    )

    assert decision.decision == AgentDecisionLabel.REJECT
    assert decision.policy_label == PolicyLabel.COUNTERFEIT
    assert "Local OCR" in decision.reason


def test_supported_ocr_exemption_is_auto_approved(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        FakeExtractor("Protective shell compatible with iPad; not made by Apple"),
    )

    decision = agent.adjudicate(
        "C112",
        ContentType.PRODUCT,
        "Product submitted for visual policy review.",
        tmp_path / "image.png",
    )

    assert decision.decision == AgentDecisionLabel.APPROVE
    assert decision.matched_policy


def test_absence_of_ocr_violation_does_not_auto_approve(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        FakeExtractor("PLAIN BLUE CANVAS TOTE NO LOGOS"),
    )

    decision = agent.adjudicate(
        "C135",
        ContentType.PRODUCT,
        "Product submitted for visual policy review.",
        tmp_path / "image.png",
    )

    assert decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert "not sufficient for approval" in decision.reason


def test_low_confidence_or_failed_ocr_routes_to_review(tmp_path: Path) -> None:
    low_confidence = build_agent(
        tmp_path,
        FakeExtractor("GUCCI 1:1 MIRROR COPY", confidence=0.2),
    )
    failed = build_agent(tmp_path, FakeExtractor(error=True))

    low_decision = low_confidence.adjudicate(
        "C072",
        ContentType.PRODUCT,
        "Visual review.",
        tmp_path / "image.png",
    )
    failed_decision = failed.adjudicate(
        "C072",
        ContentType.PRODUCT,
        "Visual review.",
        tmp_path / "image.png",
    )

    assert low_decision.decision == AgentDecisionLabel.NEED_REVIEW
    assert failed_decision.decision == AgentDecisionLabel.NEED_REVIEW

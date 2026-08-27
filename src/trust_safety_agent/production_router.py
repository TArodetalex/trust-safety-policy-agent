"""Versioned production routing across rules, OCR, LLM, and human review."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

from pydantic import Field

from trust_safety_agent.llm_client import ImageInput
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    ExemptionType,
    MatchedPolicyEvidence,
    StrictModel,
)


class ProductionRoutingConfig(StrictModel):
    routing_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    minimum_rule_confidence: float = Field(ge=0, le=1)
    minimum_ocr_decision_confidence: float = Field(ge=0, le=1)
    minimum_llm_confidence: float = Field(ge=0, le=1)
    require_image_approve_evidence: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "ProductionRoutingConfig":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class RouteStep(StrictModel):
    engine: str
    outcome: AgentDecisionLabel
    confidence: float = Field(ge=0, le=1)
    accepted: bool
    detail: str


class ProductionRouteResult(StrictModel):
    routing_version: str
    selected_engine: str
    escalated: bool
    decision: AgentDecision
    steps: List[RouteStep] = Field(min_length=1)


class RulesAgent(Protocol):
    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
    ) -> AgentDecision:
        ...


class OCRAgent(Protocol):
    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        image_path: Path,
    ) -> AgentDecision:
        ...


class MultimodalAgent(Protocol):
    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> AgentDecision:
        ...


class ProductionRouter:
    """Apply the cheapest trustworthy engine and fail closed to review."""

    def __init__(
        self,
        config: ProductionRoutingConfig,
        rules: RulesAgent,
        ocr: Optional[OCRAgent] = None,
        llm: Optional[MultimodalAgent] = None,
    ) -> None:
        self.config = config
        self.rules = rules
        self.ocr = ocr
        self.llm = llm

    @staticmethod
    def _step(
        engine: str,
        decision: AgentDecision,
        accepted: bool,
        detail: str,
    ) -> RouteStep:
        return RouteStep(
            engine=engine,
            outcome=decision.decision,
            confidence=decision.confidence,
            accepted=accepted,
            detail=detail,
        )

    @staticmethod
    def _review(
        case_id: str,
        reason: str,
        evidence: Optional[List[MatchedPolicyEvidence]] = None,
        confidence: float = 0.0,
    ) -> AgentDecision:
        return AgentDecision(
            case_id=case_id,
            decision=AgentDecisionLabel.NEED_REVIEW,
            matched_policy=evidence or [],
            reason=reason,
            recommended_action="Send the case to a human reviewer.",
            confidence=confidence,
        )

    @staticmethod
    def _result(
        routing_version: str,
        selected_engine: str,
        decision: AgentDecision,
        steps: List[RouteStep],
    ) -> ProductionRouteResult:
        return ProductionRouteResult(
            routing_version=routing_version,
            selected_engine=selected_engine,
            escalated=len(steps) > 1,
            decision=decision,
            steps=steps,
        )

    def _acceptable(
        self,
        decision: AgentDecision,
        threshold: float,
        has_image: bool,
    ) -> tuple[bool, str]:
        if decision.decision == AgentDecisionLabel.NEED_REVIEW:
            return False, "engine requested human review"
        if decision.confidence < threshold:
            return False, "decision confidence was below the routing threshold"
        if (
            has_image
            and decision.decision == AgentDecisionLabel.APPROVE
            and self.config.require_image_approve_evidence
            and not any(
                evidence.exemption_type != ExemptionType.NONE
                for evidence in decision.matched_policy
            )
        ):
            return False, "image approval lacked a specific exemption"
        return True, "decision passed production routing guards"

    @staticmethod
    def _local_image(path: Path) -> ImageInput:
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return ImageInput(data=path.read_bytes(), media_type=media_type)

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
    ) -> ProductionRouteResult:
        if image_path is not None and image_url is not None:
            decision = self._review(
                case_id,
                "Production routing rejected a case with two image sources.",
            )
            step = self._step("route_guard", decision, True, decision.reason)
            return self._result(
                self.config.routing_version,
                "human_review",
                decision,
                [step],
            )

        has_image = image_path is not None or image_url is not None
        if not has_image:
            decision = self.rules.adjudicate(
                case_id=case_id,
                content_type=content_type,
                input_text=input_text,
            )
            accepted, detail = self._acceptable(
                decision,
                self.config.minimum_rule_confidence,
                has_image=False,
            )
            step = self._step("rules", decision, accepted, detail)
            if accepted:
                return self._result(
                    self.config.routing_version,
                    "rules",
                    decision,
                    [step],
                )
            review = self._review(
                case_id,
                f"Text rules did not meet the automation contract: {decision.reason}",
                decision.matched_policy,
                decision.confidence,
            )
            return self._result(
                self.config.routing_version,
                "human_review",
                review,
                [step],
            )

        steps: List[RouteStep] = []
        latest: Optional[AgentDecision] = None
        if image_path is not None and self.ocr is not None:
            ocr_decision = self.ocr.adjudicate(
                case_id=case_id,
                content_type=content_type,
                input_text=input_text,
                image_path=image_path,
            )
            accepted, detail = self._acceptable(
                ocr_decision,
                self.config.minimum_ocr_decision_confidence,
                has_image=True,
            )
            steps.append(self._step("rules-ocr", ocr_decision, accepted, detail))
            latest = ocr_decision
            if accepted:
                return self._result(
                    self.config.routing_version,
                    "rules-ocr",
                    ocr_decision,
                    steps,
                )

        if self.llm is not None:
            try:
                image = (
                    self._local_image(image_path)
                    if image_path is not None
                    else ImageInput(url=image_url)
                )
            except (OSError, ValueError) as exc:
                review = self._review(
                    case_id,
                    f"Image preparation failed before LLM routing: {exc}",
                )
                steps.append(
                    self._step(
                        "image_guard",
                        review,
                        True,
                        "invalid or unreadable image input",
                    )
                )
                return self._result(
                    self.config.routing_version,
                    "human_review",
                    review,
                    steps,
                )

            llm_decision = self.llm.adjudicate(
                case_id=case_id,
                content_type=content_type,
                input_text=input_text,
                images=[image],
            )
            accepted, detail = self._acceptable(
                llm_decision,
                self.config.minimum_llm_confidence,
                has_image=True,
            )
            latest = llm_decision
            if (
                accepted
                and llm_decision.decision == AgentDecisionLabel.APPROVE
            ):
                steps.append(
                    self._step(
                        "multimodal-llm",
                        llm_decision,
                        False,
                        "image approval requires deterministic exemption verification",
                    )
                )
                verification = self.rules.adjudicate(
                    case_id=case_id,
                    content_type=content_type,
                    input_text=(
                        f"{input_text}\n"
                        f"Visual model observations: {llm_decision.reason}"
                    ),
                )
                verified, verification_detail = self._acceptable(
                    verification,
                    self.config.minimum_rule_confidence,
                    has_image=True,
                )
                steps.append(
                    self._step(
                        "visual-evidence-guard",
                        verification,
                        verified,
                        verification_detail,
                    )
                )
                if verified:
                    verified_decision = verification.model_copy(
                        update={
                            "confidence": min(
                                verification.confidence,
                                llm_decision.confidence,
                            ),
                            "reason": (
                                "Multimodal observations were reproduced by "
                                f"the deterministic exemption rules. "
                                f"{verification.reason}"
                            ),
                        }
                    )
                    return self._result(
                        self.config.routing_version,
                        "multimodal-llm",
                        verified_decision,
                        steps,
                    )
            else:
                steps.append(
                    self._step(
                        "multimodal-llm",
                        llm_decision,
                        accepted,
                        detail,
                    )
                )
            if accepted and llm_decision.decision != AgentDecisionLabel.APPROVE:
                return self._result(
                    self.config.routing_version,
                    "multimodal-llm",
                    llm_decision,
                    steps,
                )

        reason = (
            "No configured engine produced an image decision that passed "
            "the production automation contract."
        )
        if latest is not None:
            reason = f"{reason} Last engine result: {latest.reason}"
        review = self._review(
            case_id,
            reason,
            latest.matched_policy if latest else None,
            latest.confidence if latest else 0.0,
        )
        steps.append(
            self._step(
                "human_review",
                review,
                True,
                "fail-closed terminal route",
            )
        )
        return self._result(
            self.config.routing_version,
            "human_review",
            review,
            steps,
        )

"""Policy-grounded deterministic adjudication baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    ExemptionType,
    MatchedPolicyEvidence,
    PolicyChunk,
    PolicyLabel,
    RetrievalHit,
)
from trust_safety_agent.vector_store import PolicyVectorStore


POLICY_IDS: Dict[PolicyLabel, str] = {
    PolicyLabel.COUNTERFEIT: "POL-CF-001",
    PolicyLabel.KNOCKOFF: "POL-KO-001",
    PolicyLabel.TRADEMARK_MISUSE: "POL-TM-001",
    PolicyLabel.SHOP_IMPERSONATION: "POL-SI-001",
    PolicyLabel.RISKY_QUERY: "POL-RQ-001",
}

EXEMPTION_TITLES: Dict[ExemptionType, str] = {
    ExemptionType.COMPATIBILITY: "Compatibility",
    ExemptionType.SECOND_HAND: "Second-hand",
    ExemptionType.MEANINGFUL_WORD: "Meaningful word",
    ExemptionType.INCIDENTAL_EXPOSURE: "Incidental exposure",
    ExemptionType.CO_BRAND: "Co-brand",
}


def _pattern(*expressions: str) -> Pattern[str]:
    return re.compile("|".join(f"(?:{item})" for item in expressions), re.IGNORECASE)


COUNTERFEIT = _pattern(
    r"\bcounterfeit\b",
    r"\bfake\b",
    r"\breplica\b",
    r"\b1\s*:\s*1\b",
    r"\bmirror copy\b",
    r"\bAAA\b",
)
KNOCKOFF = _pattern(
    r"\bdupe\b",
    r"\bknockoff\b",
    r"\binspired\b.{0,30}\bcopy\b",
    r"\bsame (?:look|design) as\b",
    r"\blookalike\b",
)
QUERY_EVASION = _pattern(
    r"\bpass authenticity",
    r"\bauthenticity checks?\b",
    r"\bcustoms (?:will )?not detect\b",
    r"\bno logo photos?\b",
    r"\bbest keywords?\b",
    r"\bhid(?:e|ing) (?:counterfeit|brand)",
    r"\bevade\b",
)
SHOP_IMPERSONATION = _pattern(
    r"\bofficial (?:store|shop|outlet)\b",
    r"\bofficial\b.{0,35}\b(?:store|shop|outlet)\b",
    r"\bbrand\b.{0,35}\bofficial\b",
    r"\bauthorized (?:dealer|store|outlet)\b",
    r"\bverified flagship\b",
    r"\bflagship (?:store|shop)\b",
)
TRADEMARK_MISUSE = _pattern(
    r"\b(?:printed|print|sticker|large|full-frame).{0,35}\b(?:logo|swoosh)\b",
    r"\b(?:logo|swoosh).{0,35}\b(?:generic|unrelated)\b",
    r"\bgeneric.{0,35}\b(?:logo|swoosh)\b",
    r"\bofficial\b.{0,35}\bcollaboration\b",
)

COMPATIBILITY = _pattern(
    r"\bcompatible with\b",
    r"\bfits?\b",
    r"\bfor use with\b",
)
SECOND_HAND = _pattern(
    r"\bpre-owned\b",
    r"\bsecond-hand\b",
    r"\bused\b",
    r"\bvintage\b",
    r"\bpersonal collection\b",
)
INCIDENTAL = _pattern(
    r"\bbackground\b",
    r"\bincidental",
    r"\bface remains the main subject\b",
    r"\bvisible in the room\b",
    r"\bstreet interview\b",
    r"\bfamily photo\b",
)
CO_BRAND = _pattern(r"\b[\w-]+\s+x\s+[\w-]+\b", r"\blicensed\b")
CO_BRAND_SUPPORT = _pattern(
    r"\bproduct code\b",
    r"\boriginal tags?\b",
    r"\bgenuine\b",
    r"\bpurchased from the launch\b",
    r"\blicensed\b",
)
MEANINGFUL_WORD = _pattern(
    r"\bapple juice\b",
    r"\bgap year\b",
    r"\bshell necklace\b",
    r"\bsea shells?\b",
    r"\bamazon rainforest\b",
    r"\bpuma\b.{0,30}\bwildlife rescue\b",
)
BRAND_REFERENCE = _pattern(
    r"\bgucci\b",
    r"\bairpods?\b",
    r"\brolex\b",
    r"\bnike\b",
    r"\blv\b",
    r"\badidas\b",
    r"\bbirkin\b",
    r"\bchrome hearts\b",
    r"\bdyson\b",
    r"\bherm[eèé]s\b",
    r"\bapple\b",
    r"\bdisney\b",
    r"\bchanel\b",
    r"\bstarbucks\b",
    r"\blouis vuitton\b",
    r"\bsamsung\b",
    r"\blego\b",
    r"\bye+zys?\b",
    r"\bcoach\b",
    r"\bpatagonia\b",
    r"\bphilips\b",
    r"\bipad\b",
    r"\biphone\b",
    r"\bmacbook\b",
    r"\bcoca-cola\b",
    r"\bmcdonald'?s\b",
    r"\buniqlo\b",
    r"\bkeith haring\b",
    r"\boff-white\b",
)
COMMERCIAL_LOGO_USE = _pattern(
    r"\b(?:shirt|cap|charger|mug|tumbler|product).{0,45}\b(?:logo|swoosh)\b",
    r"\b(?:logo|swoosh).{0,45}\b(?:shirt|cap|charger|mug|tumbler|product)\b",
)


@dataclass(frozen=True)
class SignalMatch:
    policy_label: PolicyLabel
    signal: str
    confidence: float


class PolicyAdjudicator:
    """Classify one case and ground the result in the indexed policy."""

    def __init__(self, store: PolicyVectorStore, top_k: int = 5) -> None:
        self.store = store
        self.top_k = top_k

    @staticmethod
    def _exemption(text: str) -> Optional[ExemptionType]:
        if COMPATIBILITY.search(text):
            return ExemptionType.COMPATIBILITY
        if SECOND_HAND.search(text):
            return ExemptionType.SECOND_HAND
        if MEANINGFUL_WORD.search(text):
            return ExemptionType.MEANINGFUL_WORD
        if INCIDENTAL.search(text):
            return ExemptionType.INCIDENTAL_EXPOSURE
        if CO_BRAND.search(text) and CO_BRAND_SUPPORT.search(text):
            return ExemptionType.CO_BRAND
        return None

    @staticmethod
    def _violation(
        content_type: ContentType,
        text: str,
    ) -> Optional[SignalMatch]:
        has_brand = bool(BRAND_REFERENCE.search(text))
        if (
            content_type == ContentType.QUERY
            and QUERY_EVASION.search(text)
            and (COUNTERFEIT.search(text) or KNOCKOFF.search(text))
        ):
            return SignalMatch(
                PolicyLabel.RISKY_QUERY,
                "The query seeks prohibited goods while attempting to evade controls.",
                0.98,
            )
        if SHOP_IMPERSONATION.search(text) and has_brand:
            return SignalMatch(
                PolicyLabel.SHOP_IMPERSONATION,
                "The content presents a seller as an official or authorized brand shop.",
                0.96,
            )
        if COUNTERFEIT.search(text) and has_brand:
            return SignalMatch(
                PolicyLabel.COUNTERFEIT,
                "Explicit counterfeit terminology appears with a protected brand.",
                0.98,
            )
        if KNOCKOFF.search(text) and has_brand:
            return SignalMatch(
                PolicyLabel.KNOCKOFF,
                "The content explicitly promotes imitation of a named branded design.",
                0.95,
            )
        if (
            TRADEMARK_MISUSE.search(text)
            or COMMERCIAL_LOGO_USE.search(text)
        ) and has_brand:
            return SignalMatch(
                PolicyLabel.TRADEMARK_MISUSE,
                "A protected trademark is used to promote or identify an unrelated item.",
                0.92,
            )
        return None

    @staticmethod
    def _quote(chunk: PolicyChunk) -> str:
        paragraphs = [
            paragraph.strip()
            for paragraph in chunk.content.split("\n\n")
            if paragraph.strip()
        ]
        return paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]

    def _evidence_for_policy(
        self,
        text: str,
        match: SignalMatch,
        hits: Sequence[RetrievalHit],
    ) -> Optional[MatchedPolicyEvidence]:
        policy_id = POLICY_IDS[match.policy_label]
        candidate = next(
            (hit for hit in hits if hit.chunk.policy_id == policy_id),
            None,
        )
        if candidate is None:
            expanded = self.store.retrieve(
                f"{text} {match.policy_label.value.replace('_', ' ')}",
                top_k=max(self.top_k, self.store.count()),
            )
            candidate = next(
                (hit for hit in expanded if hit.chunk.policy_id == policy_id),
                None,
            )
        if candidate is None:
            return None
        return MatchedPolicyEvidence(
            policy_id=policy_id,
            chunk_id=candidate.chunk.chunk_id,
            quote=self._quote(candidate.chunk),
            retrieval_score=candidate.score,
        )

    def _exemption_evidence(
        self,
        exemption: ExemptionType,
        text: str,
        hits: Sequence[RetrievalHit],
    ) -> List[MatchedPolicyEvidence]:
        title = EXEMPTION_TITLES[exemption]
        candidate = next(
            (
                hit
                for hit in hits
                if hit.chunk.policy_id == "POL-EX-001"
                and hit.chunk.title == title
            ),
            None,
        )
        if candidate is None:
            expanded = self.store.retrieve(
                f"{text} {title} exemption",
                top_k=max(self.top_k, self.store.count()),
            )
            candidate = next(
                (
                    hit
                    for hit in expanded
                    if hit.chunk.policy_id == "POL-EX-001"
                    and hit.chunk.title == title
                ),
                None,
            )
        if candidate is None:
            return []
        return [
            MatchedPolicyEvidence(
                policy_id=candidate.chunk.policy_id,
                chunk_id=candidate.chunk.chunk_id,
                quote=self._quote(candidate.chunk),
                retrieval_score=candidate.score,
                exemption_type=exemption,
            )
        ]

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
    ) -> AgentDecision:
        text = input_text.strip()
        hits = self.store.retrieve(text, top_k=self.top_k)
        if not text:
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.NEED_REVIEW,
                reason="No content was provided for policy assessment.",
                recommended_action="Request the missing case content.",
                confidence=0.0,
            )

        violation = self._violation(content_type, text)
        exemption = self._exemption(text)

        if (
            exemption == ExemptionType.INCIDENTAL_EXPOSURE
            and violation
            and violation.policy_label == PolicyLabel.TRADEMARK_MISUSE
        ):
            violation = None

        if violation and exemption:
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.NEED_REVIEW,
                matched_policy=self._exemption_evidence(exemption, text, hits),
                reason=(
                    "The case contains both a violation signal and an exemption "
                    "signal; the available text does not resolve the conflict."
                ),
                recommended_action="Send the case to a human reviewer.",
                confidence=0.45,
            )

        if violation:
            evidence = self._evidence_for_policy(text, violation, hits)
            if evidence is None:
                return AgentDecision(
                    case_id=case_id,
                    decision=AgentDecisionLabel.NEED_REVIEW,
                    reason=(
                        f"Detected a {violation.policy_label.value} signal, but "
                        "the Policy KB did not return directly supporting evidence."
                    ),
                    recommended_action="Review the case and Policy KB coverage.",
                    confidence=0.4,
                )
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.REJECT,
                policy_label=violation.policy_label,
                matched_policy=[evidence],
                reason=violation.signal,
                recommended_action="Reject the content and record the cited policy.",
                confidence=violation.confidence,
            )

        if exemption:
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.APPROVE,
                matched_policy=self._exemption_evidence(exemption, text, hits),
                reason=(
                    f"The content matches the {exemption.value} exemption and "
                    "contains no separate violation signal."
                ),
                recommended_action="Approve the content.",
                confidence=0.94,
            )

        if BRAND_REFERENCE.search(text):
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.NEED_REVIEW,
                reason=(
                    "A brand reference is present, but the text does not establish "
                    "a violation or a supported exemption."
                ),
                recommended_action="Review context, authorization, and media evidence.",
                confidence=0.42,
            )

        return AgentDecision(
            case_id=case_id,
            decision=AgentDecisionLabel.APPROVE,
            reason="No violation signal or protected brand reference was detected.",
            recommended_action="Approve the content.",
            confidence=0.86,
        )

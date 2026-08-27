"""Multimodal LLM adjudication grounded in local policy chunks."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from pydantic import Field, ValidationError

from trust_safety_agent.llm_client import (
    ImageInput,
    LLMClientError,
    MultimodalChatClient,
)
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    ExemptionType,
    MatchedPolicyEvidence,
    PolicyChunk,
    PolicyLabel,
    RetrievalHit,
    StrictModel,
)
from trust_safety_agent.vector_store import PolicyVectorStore


MIN_AUTO_DECISION_CONFIDENCE = 0.65


class LLMDecisionDraft(StrictModel):
    decision: AgentDecisionLabel
    policy_label: Optional[PolicyLabel]
    exemption_type: ExemptionType
    evidence_chunk_ids: List[str] = Field(max_length=5)
    reason: str = Field(min_length=1, max_length=1600)
    recommended_action: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0, le=1)
    image_observations: List[str] = Field(max_length=6)
    detected_marks: List[str] = Field(max_length=10)
    uncertainties: List[str] = Field(max_length=6)


class LLMPolicyAdjudicator:
    """Use a multimodal model, then enforce policy and evidence invariants locally."""

    def __init__(
        self,
        store: PolicyVectorStore,
        client: MultimodalChatClient,
        top_k: int = 8,
        minimum_confidence: float = MIN_AUTO_DECISION_CONFIDENCE,
    ) -> None:
        self.store = store
        self.client = client
        self.top_k = top_k
        self.minimum_confidence = minimum_confidence
        self.last_schema_valid = False

    @staticmethod
    def _quote(chunk: PolicyChunk) -> str:
        paragraphs = [
            paragraph.strip()
            for paragraph in chunk.content.split("\n\n")
            if paragraph.strip()
        ]
        return paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]

    def _policy_context(
        self,
        input_text: str,
        has_images: bool,
    ) -> tuple[List[PolicyChunk], Dict[str, float]]:
        query = input_text.strip() or (
            "Visual marketplace content with products, logos, brand identity, "
            "shop identity, authenticity, imitation, and exemptions"
        )
        hits = self.store.retrieve(query, top_k=self.top_k)
        scores = {hit.chunk.chunk_id: hit.score for hit in hits}
        chunks = [hit.chunk for hit in hits]

        if has_images:
            seen = {chunk.chunk_id for chunk in chunks}
            chunks.extend(
                chunk
                for chunk in self.store.list_chunks()
                if chunk.chunk_id not in seen
            )
        return chunks, scores

    @staticmethod
    def _system_prompt(chunks: Sequence[PolicyChunk]) -> str:
        context = "\n\n".join(
            (
                f"[{chunk.chunk_id}] policy_id={chunk.policy_id}; "
                f"policy_label="
                f"{chunk.policy_label.value if chunk.policy_label else 'none'}; "
                f"exemption={chunk.exemption_type.value}; title={chunk.title}\n"
                f"{chunk.content}"
            )
            for chunk in chunks
        )
        return f"""You are a Trust & Safety policy adjudicator.
Judge only from the supplied case and POLICY CONTEXT. Inspect every supplied
image carefully for visible text, logos, product identity, shop identity,
counterfeit indicators, imitation claims, and exemption context.

Rules:
1. Return reject only when a supplied policy directly supports the violation.
2. Return approve when an exemption clearly applies or no violation is present.
3. Return need_review when evidence is missing, ambiguous, conflicting, or the
   image is unreadable.
4. Never invent policy IDs or chunk IDs. evidence_chunk_ids must contain only
   IDs shown in POLICY CONTEXT.
5. policy_label must be null unless decision is reject.
6. Treat visual observations as evidence, but do not infer authenticity from a
   logo, low price, visual similarity, or brand presence alone.

Return one JSON object with exactly these keys:
decision, policy_label, exemption_type, evidence_chunk_ids, reason,
recommended_action, confidence, image_observations, detected_marks,
uncertainties.

Allowed decision values: approve, reject, need_review.
Allowed policy_label values: counterfeit, knockoff, trademark_misuse,
shop_impersonation, risky_query, or null.
Allowed exemption_type values: none, compatibility, second_hand,
meaningful_word, incidental_exposure, co_brand.
confidence must be between 0 and 1. image_observations must be short factual
statements about visible evidence, or an empty list when no image is supplied.
detected_marks lists visible brand or logo text only. uncertainties lists
material ambiguity, occlusion, unreadable regions, or an empty list.

POLICY CONTEXT
{context}"""

    @staticmethod
    def _user_prompt(
        content_type: ContentType,
        input_text: str,
        image_count: int,
    ) -> str:
        payload = {
            "content_type": content_type.value,
            "text": input_text.strip(),
            "attached_images": image_count,
        }
        return (
            "Assess this case. Base the decision on the combined text and image "
            f"evidence.\nCASE\n{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _review_decision(
        case_id: str,
        reason: str,
        confidence: float = 0.0,
        evidence: Optional[List[MatchedPolicyEvidence]] = None,
    ) -> AgentDecision:
        return AgentDecision(
            case_id=case_id,
            decision=AgentDecisionLabel.NEED_REVIEW,
            matched_policy=evidence or [],
            reason=reason[:2000],
            recommended_action="Send the case to a human reviewer.",
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _normalize(
        self,
        case_id: str,
        draft: LLMDecisionDraft,
        chunks: Sequence[PolicyChunk],
        scores: Dict[str, float],
    ) -> AgentDecision:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        selected_chunks = [
            chunk_map[chunk_id]
            for chunk_id in draft.evidence_chunk_ids
            if chunk_id in chunk_map
        ]
        evidence = [
            MatchedPolicyEvidence(
                policy_id=chunk.policy_id,
                chunk_id=chunk.chunk_id,
                quote=self._quote(chunk),
                retrieval_score=scores.get(chunk.chunk_id),
                exemption_type=chunk.exemption_type,
            )
            for chunk in selected_chunks
        ]

        reason = draft.reason
        if draft.image_observations:
            observations = "; ".join(draft.image_observations)
            reason = f"{reason} Image observations: {observations}"
        if draft.detected_marks:
            reason = f"{reason} Detected marks: {'; '.join(draft.detected_marks)}"
        if draft.uncertainties:
            reason = f"{reason} Uncertainties: {'; '.join(draft.uncertainties)}"

        if draft.confidence < self.minimum_confidence:
            return self._review_decision(
                case_id,
                f"Model confidence was below the automation threshold. {reason}",
                draft.confidence,
                evidence,
            )

        if draft.decision == AgentDecisionLabel.REJECT:
            if draft.exemption_type != ExemptionType.NONE:
                return self._review_decision(
                    case_id,
                    "The model returned an exemption for a reject decision.",
                    draft.confidence,
                    evidence,
                )
            if draft.policy_label is None:
                return self._review_decision(
                    case_id,
                    "The model proposed rejection without a policy label.",
                    draft.confidence,
                    evidence,
                )
            supported = [
                item
                for item, chunk in zip(evidence, selected_chunks)
                if chunk.policy_label == draft.policy_label
            ]
            if not supported:
                return self._review_decision(
                    case_id,
                    "The model proposed rejection without a valid matching policy chunk.",
                    draft.confidence,
                    evidence,
                )
            return AgentDecision(
                case_id=case_id,
                decision=AgentDecisionLabel.REJECT,
                policy_label=draft.policy_label,
                matched_policy=supported,
                reason=reason,
                recommended_action=draft.recommended_action,
                confidence=draft.confidence,
            )

        if draft.policy_label is not None:
            return self._review_decision(
                case_id,
                "The model returned a policy label for a non-reject decision.",
                draft.confidence,
                evidence,
            )
        if draft.decision != AgentDecisionLabel.APPROVE:
            if draft.exemption_type != ExemptionType.NONE:
                return self._review_decision(
                    case_id,
                    "The model returned an exemption for a review decision.",
                    draft.confidence,
                    evidence,
                )
        elif draft.exemption_type != ExemptionType.NONE:
            supported = [
                item
                for item, chunk in zip(evidence, selected_chunks)
                if chunk.exemption_type == draft.exemption_type
            ]
            if not supported:
                return self._review_decision(
                    case_id,
                    "The model proposed an exemption without a matching policy chunk.",
                    draft.confidence,
                    evidence,
                )
            evidence = supported

        return AgentDecision(
            case_id=case_id,
            decision=draft.decision,
            matched_policy=evidence,
            reason=reason,
            recommended_action=draft.recommended_action,
            confidence=draft.confidence,
        )

    def adjudicate(
        self,
        case_id: str,
        content_type: ContentType,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> AgentDecision:
        if not input_text.strip() and not images:
            return self._review_decision(
                case_id,
                "No text or image was provided for policy assessment.",
            )
        chunks, scores = self._policy_context(input_text, bool(images))
        if not chunks:
            return self._review_decision(
                case_id,
                "The Policy KB is empty, so the case cannot be grounded.",
            )

        try:
            self.last_schema_valid = False
            payload = self.client.complete_json(
                system_prompt=self._system_prompt(chunks),
                user_text=self._user_prompt(
                    content_type,
                    input_text,
                    len(images),
                ),
                images=images,
                response_schema=LLMDecisionDraft.model_json_schema(),
                schema_name="trust_safety_decision_v2",
            )
            draft = LLMDecisionDraft.model_validate(payload)
            self.last_schema_valid = True
        except (LLMClientError, ValidationError, ValueError) as exc:
            return self._review_decision(
                case_id,
                f"Multimodal LLM inference could not be validated: {exc}",
            )
        return self._normalize(case_id, draft, chunks, scores)

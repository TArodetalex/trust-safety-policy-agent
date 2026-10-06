"""Observable Phase 9 Shop and Product review workflows."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from trust_safety_agent.brand_knowledge import BrandKnowledgeStore
from trust_safety_agent.brand_library import ControlledBrandLibrary, normalize_brand_text
from trust_safety_agent.case_store import (
    AgentRunRecord,
    JsonlStore,
    LabelStatus,
    ProductCase,
    RunDecision,
    ShopCase,
)
from trust_safety_agent.price_search import (
    PriceSearchProvider,
    PriceSearchStatus,
    ReferencePriceResult,
)
from trust_safety_agent.schema import StrictModel, StringEnum
from trust_safety_agent.trace import (
    ExecutionTrace,
    TraceNode,
    TraceNodeStatus,
    TraceStore,
    sanitize_sensitive,
)
from trust_safety_agent.vector_store import PolicyVectorStore


WORKFLOW_VERSION = "v9.1.0"
POLICY_INDEX_VERSION = "synthetic-policy-v2"
BRAND_INDEX_VERSION = "v1.0.0"


class BrandControlStatus(StringEnum):
    CONTROLLED = "controlled"
    UNCONTROLLED = "uncontrolled"
    UNKNOWN = "unknown"


class ShopItemStatus(StringEnum):
    VIOLATION = "violation"
    COMPLIANT = "compliant"
    MANUAL_REVIEW = "manual_review"


class ProductRuntimeEvidence(StrictModel):
    detected_brand: Optional[str] = None
    detected_product: str = ""
    detected_brand_source: str = "model_or_fixture"
    detected_brand_confidence: float = Field(default=0.8, ge=0, le=1)
    observed_product_brand: Optional[str] = None
    logo_match: str = Field(default="none", pattern=r"^(exact|modified|none|unclear)$")
    distinctive_design: bool = False
    white_label: bool = False
    image_readable: bool = True
    fake_risk_evidence: bool = False
    visual_description: str = ""


class ShopRuntimeEvidence(StrictModel):
    detected_name_brand: Optional[str] = None
    detected_avatar_brand: Optional[str] = None
    avatar_readable: bool = True
    avatar_confidence: float = Field(default=0.8, ge=0, le=1)
    official_identity_signal: bool = False
    visual_description: str = ""


class LabelAssessment(StrictModel):
    status: LabelStatus
    reason: str
    evidence: List[str] = Field(default_factory=list)
    qualifiers: List[str] = Field(default_factory=list)


class ProductWorkflowResult(StrictModel):
    run_id: str
    case_id: str
    detected_brand: Optional[str]
    detected_brand_source: Optional[str]
    detected_brand_confidence: float = Field(ge=0, le=1)
    brand_control_status: BrandControlStatus
    labels: Dict[str, LabelAssessment]
    suggested_decision: RunDecision
    confidence: float = Field(ge=0, le=1)
    notices: List[str] = Field(default_factory=list)
    guardrail_triggered: bool = False
    trace_id: str
    price_evidence: Optional[ReferencePriceResult] = None


class ShopWorkflowResult(StrictModel):
    run_id: str
    case_id: str
    detected_brand: Optional[str]
    brand_control_status: BrandControlStatus
    shop_name_status: ShopItemStatus
    shop_avatar_status: ShopItemStatus
    suggested_decision: RunDecision
    confidence: float = Field(ge=0, le=1)
    notices: List[str] = Field(default_factory=list)
    trace_id: str


class ReviewSession(StrictModel):
    session_id: str = Field(default_factory=lambda: f"SES-{uuid4().hex[:12].upper()}")
    route_run_id: Optional[str] = None
    route_trace_id: Optional[str] = None
    shop_result: Optional[ShopWorkflowResult] = None
    product_result: Optional[ProductWorkflowResult] = None


class WorkflowTraceRecorder:
    def __init__(
        self,
        *,
        run_id: str,
        case_id: str,
        case_revision: int,
        workflow: str,
        prompt_version: str,
        skill_version: str,
        model_id: str,
        parent_run_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.case_revision = case_revision
        self.workflow = workflow
        self.prompt_version = prompt_version
        self.skill_version = skill_version
        self.model_id = model_id
        self.parent_run_id = parent_run_id
        self.nodes: List[TraceNode] = []
        self.trace_id = f"TRC-{uuid4().hex[:12].upper()}"

    def execute(
        self,
        node_id: str,
        input_summary: Dict[str, Any],
        action: Callable[[], Dict[str, Any]],
        *,
        candidate_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            output = action()
            status = TraceNodeStatus.SUCCESS
            error = None
        except Exception as exc:
            output = {}
            status = TraceNodeStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
        ended_at = datetime.now(timezone.utc)
        self.nodes.append(TraceNode(
            node_name=node_id,
            node_id=node_id,
            node_version="v1.0.0",
            started_at=started_at,
            ended_at=ended_at,
            input_summary=sanitize_sensitive(input_summary),
            output=sanitize_sensitive(output),
            candidate_evidence=sanitize_sensitive(candidate_evidence or []),
            status=status,
            latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
            error=error,
        ))
        if error:
            raise RuntimeError(error)
        return output

    def skipped(self, node_id: str, reason: str) -> None:
        now = datetime.now(timezone.utc)
        self.nodes.append(TraceNode(
            node_name=node_id,
            node_id=node_id,
            started_at=now,
            ended_at=now,
            input_summary={},
            output={"reason": reason},
            status=TraceNodeStatus.SKIPPED,
            latency_ms=0,
        ))

    def finish(self, final_output: Dict[str, Any]) -> ExecutionTrace:
        return ExecutionTrace(
            trace_id=self.trace_id,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            case_id=self.case_id,
            case_revision=self.case_revision,
            workflow=self.workflow,
            workflow_version=WORKFLOW_VERSION,
            prompt_version=self.prompt_version,
            skill_version=self.skill_version,
            model_id=self.model_id,
            policy_index_version=POLICY_INDEX_VERSION,
            brand_index_version=BRAND_INDEX_VERSION,
            nodes=self.nodes,
            final_output=sanitize_sensitive(final_output),
        )


def _compatibility(text: str) -> bool:
    return bool(re.search(r"\b(compatible with|fits|for use with)\b|兼容|适用于", text, re.I))


def _fake_wording(text: str) -> bool:
    return bool(re.search(r"\b(fake|counterfeit|replica|mirror copy|1:1)\b|假货|高仿|复刻", text, re.I))


def aggregate_product_labels(labels: Dict[str, LabelAssessment]) -> tuple[RunDecision, bool]:
    conflict = (
        labels.get("Counterfeit", LabelAssessment(status=LabelStatus.NOT_HIT, reason="missing")).status == LabelStatus.HIT
        and labels.get("Knockoff", LabelAssessment(status=LabelStatus.NOT_HIT, reason="missing")).status == LabelStatus.HIT
    )
    if conflict:
        return RunDecision.MANUAL_REVIEW, True
    statuses = [item.status for item in labels.values()]
    if LabelStatus.HIT in statuses:
        return RunDecision.REJECT, False
    if LabelStatus.INSUFFICIENT_EVIDENCE in statuses:
        return RunDecision.MANUAL_REVIEW, False
    return RunDecision.APPROVE, False


class ProductWorkflow:
    def __init__(
        self,
        policy_store: PolicyVectorStore,
        brand_store: BrandKnowledgeStore,
        brand_library: ControlledBrandLibrary,
        price_provider: PriceSearchProvider,
        trace_store: Optional[TraceStore] = None,
        run_store: Optional[JsonlStore] = None,
        *,
        model_id: str = "offline-rules-v1",
        prompt_version: str = "PRM-PRODUCT-IPR@v9.1.0",
        skill_version: str = "SKL-PRODUCT-IPR@v9.1.0",
    ) -> None:
        self.policy_store = policy_store
        self.brand_store = brand_store
        self.brand_library = brand_library
        self.price_provider = price_provider
        self.trace_store = trace_store
        self.run_store = run_store
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.skill_version = skill_version

    def run(self, case: ProductCase, evidence: Optional[ProductRuntimeEvidence] = None, *, parent_run_id: Optional[str] = None) -> tuple[ProductWorkflowResult, AgentRunRecord, ExecutionTrace]:
        evidence = evidence or ProductRuntimeEvidence()
        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        recorder = WorkflowTraceRecorder(run_id=run_id, case_id=case.product_id, case_revision=case.revision, workflow="product_ipr", prompt_version=self.prompt_version, skill_version=self.skill_version, model_id=self.model_id, parent_run_id=parent_run_id)

        recorder.execute("validate_input", {"product_id": case.product_id}, lambda: {"valid": bool(case.title and case.product_images), "image_count": len(case.product_images)})
        combined_text = f"{case.title}\n{case.description}"
        normalized = recorder.execute("normalize_and_extract_evidence", {"title": case.title, "image_count": len(case.product_images)}, lambda: {"normalized_text": normalize_brand_text(combined_text), "image_readable": evidence.image_readable, "visual_description": evidence.visual_description})
        mentions = self.brand_library.find_mentions(combined_text)
        detected_brand = evidence.detected_brand or (mentions[0].brand_name if mentions else case.brand)
        controlled = self.brand_library.get(detected_brand or "") is not None
        control_status = BrandControlStatus.CONTROLLED if controlled else (BrandControlStatus.UNCONTROLLED if detected_brand else BrandControlStatus.UNKNOWN)
        recorder.execute("recall_entities", {"raw_brand": case.brand, "text": combined_text[:500]}, lambda: {"detected_brand": detected_brand, "detected_brand_source": evidence.detected_brand_source if evidence.detected_brand else ("text" if mentions else "raw_brand"), "confidence": evidence.detected_brand_confidence, "brand_control_status": control_status.value})

        policy_hits = self.policy_store.retrieve(f"Product IPR {combined_text} exemptions evidence", top_k=6)
        policy_candidates = [{"chunk_id": hit.chunk.chunk_id, "score": hit.score} for hit in policy_hits]
        recorder.execute("retrieve_policy", {"query": combined_text[:500], "index_version": POLICY_INDEX_VERSION}, lambda: {"candidate_chunk_ids": [item["chunk_id"] for item in policy_candidates], "candidate_scores": [item["score"] for item in policy_candidates], "selected_chunk_ids": [item["chunk_id"] for item in policy_candidates[:3]], "filters": {"workflow": "product_ipr"}}, candidate_evidence=policy_candidates)

        brand_hits = self.brand_store.retrieve(f"{detected_brand or ''} {evidence.detected_product} {combined_text}", top_k=5)
        brand_candidates = [{"chunk_id": hit.chunk.chunk_id, "brand": hit.chunk.brand_name, "score": hit.score} for hit in brand_hits]
        recorder.execute("retrieve_brand_knowledge", {"query": f"{detected_brand or ''} {evidence.detected_product}", "index_version": BRAND_INDEX_VERSION}, lambda: {"candidate_chunk_ids": [item["chunk_id"] for item in brand_candidates], "candidate_scores": [item["score"] for item in brand_candidates], "selected_chunk_ids": [item["chunk_id"] for item in brand_candidates[:2]], "filters": {"entity_only": True}}, candidate_evidence=brand_candidates)

        price_evidence: Optional[ReferencePriceResult] = None
        labels: Dict[str, LabelAssessment] = {}

        price_needed = evidence.logo_match == "exact" and (evidence.fake_risk_evidence or _fake_wording(combined_text))
        if price_needed:
            price_output = recorder.execute("retrieve_reference_price", {"detected_brand": detected_brand, "detected_product": evidence.detected_product, "market": "US"}, lambda: self.price_provider.search_reference_price(detected_brand or "", evidence.detected_product).model_dump(mode="json"))
            price_evidence = ReferencePriceResult.model_validate(price_output)
        else:
            recorder.skipped("retrieve_reference_price", "Counterfeit price evidence was not required")

        def evaluate_rules() -> Dict[str, Any]:
            nonlocal labels
            fake_risk = evidence.fake_risk_evidence or _fake_wording(combined_text)
            if not evidence.image_readable or evidence.logo_match == "unclear":
                counterfeit = LabelAssessment(status=LabelStatus.INSUFFICIENT_EVIDENCE, reason="图片或 Logo 不清晰，无法确认精确商标。")
            elif evidence.logo_match == "exact" and fake_risk:
                if price_evidence and price_evidence.status == PriceSearchStatus.SUCCESS and case.price is not None and price_evidence.reference_price_selected is not None:
                    ratio = case.price / price_evidence.reference_price_selected
                    counterfeit = LabelAssessment(status=LabelStatus.HIT if ratio < 0.3 else LabelStatus.NOT_HIT, reason=f"商品价格为可靠参考价的 {ratio:.1%}。", evidence=[source.url for source in price_evidence.sources])
                else:
                    counterfeit = LabelAssessment(status=LabelStatus.INSUFFICIENT_EVIDENCE, reason="存在精确 Logo 和假货风险信号，但缺少可靠参考价。")
            else:
                counterfeit = LabelAssessment(status=LabelStatus.NOT_HIT, reason="未同时满足精确 Logo、假货风险证据和可靠低价阈值。")

            if counterfeit.status == LabelStatus.HIT:
                knockoff = LabelAssessment(status=LabelStatus.NOT_HIT, reason="Counterfeit 已命中，按互斥规则不再命中 Knockoff。")
            elif not evidence.image_readable or evidence.logo_match == "unclear":
                knockoff = LabelAssessment(status=LabelStatus.INSUFFICIENT_EVIDENCE, reason="图片证据不足，无法区分变形 Logo 或外观仿冒。")
            elif evidence.logo_match == "modified":
                knockoff = LabelAssessment(status=LabelStatus.HIT, reason="检测到明显变形或近似 Logo。", qualifiers=["modified_logo"])
            elif evidence.distinctive_design:
                subtype = "white_label_lookalike" if evidence.white_label else "distinctive_design"
                knockoff = LabelAssessment(status=LabelStatus.HIT, reason="检测到具有辨识度的非通用设计模仿。", qualifiers=[subtype])
            else:
                knockoff = LabelAssessment(status=LabelStatus.NOT_HIT, reason="未发现变形 Logo 或具有辨识度的设计模仿。")

            raw_brand = case.brand or ""
            raw_controlled = self.brand_library.get(raw_brand) is not None
            raw_mentions = [item for item in mentions if normalize_brand_text(item.brand_name) == normalize_brand_text(raw_brand)]
            if not raw_brand:
                mba = LabelAssessment(status=LabelStatus.NOT_APPLICABLE, reason="原始 Brand 为空，MBA 不适用。")
            elif not raw_controlled:
                mba = LabelAssessment(status=LabelStatus.NOT_APPLICABLE, reason="原始 Brand 不在公共或合成管控库中，MBA 不适用。")
            elif _compatibility(combined_text):
                mba = LabelAssessment(status=LabelStatus.NOT_HIT, reason="品牌提及符合兼容性豁免。", qualifiers=["compatibility_exemption"])
            elif case.brand_authorized is True:
                mba = LabelAssessment(status=LabelStatus.NOT_HIT, reason="原始 Brand 已通过授权校验。")
            elif raw_mentions or (evidence.logo_match == "exact" and normalize_brand_text(evidence.detected_brand or "") == normalize_brand_text(raw_brand)):
                qualifiers = ["provisional_missing_authorization"] if case.brand_authorized is None else []
                mba = LabelAssessment(status=LabelStatus.HIT, reason="管控品牌存在明确指示器且授权未通过。", qualifiers=qualifiers)
            else:
                mba = LabelAssessment(status=LabelStatus.NOT_HIT, reason="未发现与原始 Brand 完全一致的品牌指示器。")

            info_brands = {item.brand_name for item in mentions}
            info_controlled = {name for name in info_brands if self.brand_library.get(name)}
            if not info_controlled:
                tmi = LabelAssessment(status=LabelStatus.NOT_APPLICABLE, reason="信息层没有指向管控品牌，TMI 不适用。")
            elif _compatibility(combined_text):
                tmi = LabelAssessment(status=LabelStatus.NOT_HIT, reason="信息层品牌提及符合兼容性豁免。", qualifiers=["compatibility_exemption"])
            elif not evidence.image_readable:
                tmi = LabelAssessment(status=LabelStatus.INSUFFICIENT_EVIDENCE, reason="商品实物图片不可读，无法比较信息层与实物层。")
            elif not evidence.observed_product_brand:
                tmi = LabelAssessment(status=LabelStatus.INSUFFICIENT_EVIDENCE, reason="缺少商品实物品牌识别结果。")
            elif evidence.observed_product_brand.casefold() == "unbranded" or all(normalize_brand_text(evidence.observed_product_brand) != normalize_brand_text(name) for name in info_controlled):
                tmi = LabelAssessment(status=LabelStatus.HIT, reason="信息层管控品牌与商品实物品牌不一致。", evidence=[f"信息层: {', '.join(sorted(info_controlled))}", f"实物层: {evidence.observed_product_brand}"])
            else:
                tmi = LabelAssessment(status=LabelStatus.NOT_HIT, reason="信息层与商品实物层品牌一致。")
            labels = {"Counterfeit": counterfeit, "Knockoff": knockoff, "MBA": mba, "TMI": tmi}
            return {name: assessment.model_dump(mode="json") for name, assessment in labels.items()}

        recorder.execute("evaluate_rules", {"detected_brand": detected_brand, "brand_control_status": control_status.value, "normalized_evidence": normalized}, evaluate_rules)

        def aggregate() -> Dict[str, Any]:
            decision, conflict = aggregate_product_labels(labels)
            return {"suggested_decision": decision.value, "hit_labels": [name for name, value in labels.items() if value.status == LabelStatus.HIT], "conflict": conflict}

        aggregate_output = recorder.execute("aggregate_decision", {"label_statuses": {name: value.status.value for name, value in labels.items()}}, aggregate)
        decision = RunDecision(aggregate_output["suggested_decision"])
        conflict = bool(aggregate_output["conflict"])
        guardrail = recorder.execute("apply_guardrail", {"counterfeit": labels["Counterfeit"].status.value, "knockoff": labels["Knockoff"].status.value}, lambda: {"conflict": conflict, "action": "manual_review" if conflict else "pass"})
        if guardrail["conflict"]:
            decision = RunDecision.MANUAL_REVIEW

        notices = []
        if case.brand_authorized is None and case.brand:
            notices.append("请补充品牌授权情况。")
        confidence = 0.35 if decision == RunDecision.MANUAL_REVIEW else min(0.98, max(0.55, evidence.detected_brand_confidence))
        final = {"suggested_decision": decision.value, "labels": {name: value.model_dump(mode="json") for name, value in labels.items()}, "notices": notices, "guardrail_triggered": bool(guardrail["conflict"])}
        recorder.execute("publish_result", {"guardrail_passed": not guardrail["conflict"]}, lambda: final)
        trace = recorder.finish(final)
        if self.trace_store:
            self.trace_store.append(trace)
        result = ProductWorkflowResult(run_id=run_id, case_id=case.product_id, detected_brand=detected_brand, detected_brand_source=evidence.detected_brand_source if detected_brand else None, detected_brand_confidence=evidence.detected_brand_confidence, brand_control_status=control_status, labels=labels, suggested_decision=decision, confidence=confidence, notices=notices, guardrail_triggered=bool(guardrail["conflict"]), trace_id=trace.trace_id, price_evidence=price_evidence)
        run = AgentRunRecord(run_id=run_id, case_id=case.product_id, case_revision=case.revision, workflow_version=WORKFLOW_VERSION, prompt_version=self.prompt_version, skill_version=self.skill_version, model_id=self.model_id, policy_index_version=POLICY_INDEX_VERSION, brand_index_version=BRAND_INDEX_VERSION, suggested_labels={name: item.status for name, item in labels.items()}, suggested_decision=decision, confidence=confidence, trace_id=trace.trace_id, parent_run_id=parent_run_id)
        if self.run_store:
            self.run_store.append(run)
        return result, run, trace


class ShopWorkflow:
    def __init__(self, policy_store: PolicyVectorStore, brand_store: BrandKnowledgeStore, brand_library: ControlledBrandLibrary, trace_store: Optional[TraceStore] = None, run_store: Optional[JsonlStore] = None, *, model_id: str = "offline-rules-v1", prompt_version: str = "PRM-SHOP@v9.1.0", skill_version: str = "SKL-SHOP@v9.1.0") -> None:
        self.policy_store = policy_store
        self.brand_store = brand_store
        self.brand_library = brand_library
        self.trace_store = trace_store
        self.run_store = run_store
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.skill_version = skill_version

    def run(self, case: ShopCase, evidence: Optional[ShopRuntimeEvidence] = None, *, parent_run_id: Optional[str] = None) -> tuple[ShopWorkflowResult, AgentRunRecord, ExecutionTrace]:
        evidence = evidence or ShopRuntimeEvidence()
        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        recorder = WorkflowTraceRecorder(run_id=run_id, case_id=case.shop_id, case_revision=case.revision, workflow="shop_identity", prompt_version=self.prompt_version, skill_version=self.skill_version, model_id=self.model_id, parent_run_id=parent_run_id)
        recorder.execute("validate_input", {"shop_id": case.shop_id}, lambda: {"valid": bool(case.shop_name and case.shop_avatar)})
        recorder.execute("normalize_and_extract_evidence", {"shop_name": case.shop_name, "avatar": case.shop_avatar}, lambda: {"normalized_name": normalize_brand_text(case.shop_name), "avatar_readable": evidence.avatar_readable, "visual_description": evidence.visual_description})
        mentions = self.brand_library.find_mentions(case.shop_name)
        detected = evidence.detected_name_brand or evidence.detected_avatar_brand or (mentions[0].brand_name if mentions else case.brand)
        controlled = self.brand_library.get(detected or "") is not None
        control_status = BrandControlStatus.CONTROLLED if controlled else (BrandControlStatus.UNCONTROLLED if detected else BrandControlStatus.UNKNOWN)
        recorder.execute("recall_entities", {"raw_brand": case.brand}, lambda: {"detected_brand": detected, "brand_control_status": control_status.value})
        policy_hits = self.policy_store.retrieve(f"shop name avatar authorization meaningful word {case.shop_name}", top_k=5)
        recorder.execute("retrieve_policy", {"query": case.shop_name, "index_version": POLICY_INDEX_VERSION}, lambda: {"candidate_chunk_ids": [hit.chunk.chunk_id for hit in policy_hits], "candidate_scores": [hit.score for hit in policy_hits], "selected_chunk_ids": [hit.chunk.chunk_id for hit in policy_hits[:3]], "filters": {"workflow": "shop_identity"}}, candidate_evidence=[{"chunk_id": hit.chunk.chunk_id, "score": hit.score} for hit in policy_hits])
        brand_hits = self.brand_store.retrieve(f"{detected or ''} {case.shop_name}", top_k=4)
        recorder.execute("retrieve_brand_knowledge", {"query": f"{detected or ''} {case.shop_name}", "index_version": BRAND_INDEX_VERSION}, lambda: {"candidate_chunk_ids": [hit.chunk.chunk_id for hit in brand_hits], "candidate_scores": [hit.score for hit in brand_hits], "selected_chunk_ids": [hit.chunk.chunk_id for hit in brand_hits[:2]], "filters": {"entity_only": True}}, candidate_evidence=[{"chunk_id": hit.chunk.chunk_id, "score": hit.score} for hit in brand_hits])

        def is_authorized(target: Optional[str]) -> bool:
            return bool(target and case.brand and case.brand_authorized is True and normalize_brand_text(target) == normalize_brand_text(case.brand))

        brand = self.brand_library.get(detected or "")
        common_word = bool(brand and brand.is_common_word)
        name_has_official = bool(re.search(r"\b(official|flagship|authorized|store)\b|官方|旗舰|授权|专卖", case.shop_name, re.I)) or evidence.official_identity_signal

        def evaluate() -> Dict[str, Any]:
            if detected and controlled and not is_authorized(evidence.detected_name_brand or detected) and name_has_official:
                name_status = ShopItemStatus.VIOLATION
            elif common_word and not name_has_official:
                name_status = ShopItemStatus.MANUAL_REVIEW if len(case.shop_name.split()) <= 1 else ShopItemStatus.COMPLIANT
            elif detected and controlled and not is_authorized(evidence.detected_name_brand or detected):
                name_status = ShopItemStatus.MANUAL_REVIEW
            else:
                name_status = ShopItemStatus.COMPLIANT

            avatar_brand = evidence.detected_avatar_brand
            if not evidence.avatar_readable or evidence.avatar_confidence < 0.5:
                avatar_status = ShopItemStatus.MANUAL_REVIEW
            elif avatar_brand and self.brand_library.get(avatar_brand) and not is_authorized(avatar_brand):
                avatar_status = ShopItemStatus.VIOLATION
            else:
                avatar_status = ShopItemStatus.COMPLIANT
            return {"shop_name_status": name_status.value, "shop_avatar_status": avatar_status.value}

        assessed = recorder.execute("evaluate_rules", {"detected_brand": detected, "brand_authorized": case.brand_authorized}, evaluate)
        name_status = ShopItemStatus(assessed["shop_name_status"])
        avatar_status = ShopItemStatus(assessed["shop_avatar_status"])
        decision = RunDecision.REJECT if ShopItemStatus.VIOLATION in {name_status, avatar_status} else (RunDecision.MANUAL_REVIEW if ShopItemStatus.MANUAL_REVIEW in {name_status, avatar_status} else RunDecision.APPROVE)
        recorder.execute("aggregate_decision", assessed, lambda: {"suggested_decision": decision.value})
        recorder.execute("apply_guardrail", {"avatar_confidence": evidence.avatar_confidence}, lambda: {"action": "manual_review" if decision == RunDecision.MANUAL_REVIEW else "pass"})
        notices = ["请补充品牌授权情况。"] if case.brand and case.brand_authorized is None else []
        final = {"shop_name_status": name_status.value, "shop_avatar_status": avatar_status.value, "suggested_decision": decision.value, "notices": notices}
        recorder.execute("publish_result", {"decision": decision.value}, lambda: final)
        trace = recorder.finish(final)
        if self.trace_store:
            self.trace_store.append(trace)
        confidence = 0.4 if decision == RunDecision.MANUAL_REVIEW else max(0.6, evidence.avatar_confidence)
        result = ShopWorkflowResult(run_id=run_id, case_id=case.shop_id, detected_brand=detected, brand_control_status=control_status, shop_name_status=name_status, shop_avatar_status=avatar_status, suggested_decision=decision, confidence=confidence, notices=notices, trace_id=trace.trace_id)
        statuses = {"shop_name": LabelStatus.HIT if name_status == ShopItemStatus.VIOLATION else (LabelStatus.INSUFFICIENT_EVIDENCE if name_status == ShopItemStatus.MANUAL_REVIEW else LabelStatus.NOT_HIT), "shop_avatar": LabelStatus.HIT if avatar_status == ShopItemStatus.VIOLATION else (LabelStatus.INSUFFICIENT_EVIDENCE if avatar_status == ShopItemStatus.MANUAL_REVIEW else LabelStatus.NOT_HIT)}
        run = AgentRunRecord(run_id=run_id, case_id=case.shop_id, case_revision=case.revision, workflow_version=WORKFLOW_VERSION, prompt_version=self.prompt_version, skill_version=self.skill_version, model_id=self.model_id, policy_index_version=POLICY_INDEX_VERSION, brand_index_version=BRAND_INDEX_VERSION, suggested_labels=statuses, suggested_decision=decision, confidence=confidence, trace_id=trace.trace_id, parent_run_id=parent_run_id)
        if self.run_store:
            self.run_store.append(run)
        return result, run, trace


class ReviewRouter:
    def __init__(self, shop_workflow: ShopWorkflow, product_workflow: ProductWorkflow) -> None:
        self.shop_workflow = shop_workflow
        self.product_workflow = product_workflow

    def run(self, *, shop_case: Optional[ShopCase] = None, product_case: Optional[ProductCase] = None, shop_evidence: Optional[ShopRuntimeEvidence] = None, product_evidence: Optional[ProductRuntimeEvidence] = None) -> ReviewSession:
        if shop_case is None and product_case is None:
            raise ValueError("at least one Shop Case or Product Case is required")
        route_run_id = f"RUN-{uuid4().hex[:12].upper()}"
        route_case_id = "+".join(item for item in [shop_case.shop_id if shop_case else "", product_case.product_id if product_case else ""] if item)
        recorder = WorkflowTraceRecorder(
            run_id=route_run_id,
            case_id=route_case_id,
            case_revision=1,
            workflow="comprehensive_router",
            prompt_version="deterministic-router-v1",
            skill_version="route-workflow-v1",
            model_id="deterministic-router",
        )
        route_output = recorder.execute(
            "route_workflow",
            {"has_shop": shop_case is not None, "has_product": product_case is not None},
            lambda: {"subtasks": [name for name, present in (("shop_identity", shop_case is not None), ("product_ipr", product_case is not None)) if present]},
        )
        route_trace = recorder.finish(route_output)
        trace_store = self.shop_workflow.trace_store or self.product_workflow.trace_store
        if trace_store:
            trace_store.append(route_trace)
        session = ReviewSession(route_run_id=route_run_id, route_trace_id=route_trace.trace_id)
        if shop_case:
            session.shop_result = self.shop_workflow.run(shop_case, shop_evidence, parent_run_id=route_run_id)[0]
        if product_case:
            session.product_result = self.product_workflow.run(product_case, product_evidence, parent_run_id=route_run_id)[0]
        return session

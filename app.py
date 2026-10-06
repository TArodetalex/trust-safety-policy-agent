"""Streamlit workspace for policy and product-review workflows."""

from html import escape
from pathlib import Path
import time
from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.batch_io import (
    BatchValidationError,
    export_csv,
    export_xlsx,
    parse_product_batch,
    parse_shop_batch,
)
from trust_safety_agent.brand_library import ControlledBrandLibrary
from trust_safety_agent.config import (
    DEFAULT_AGENT_RUNS_PATH,
    DEFAULT_BAD_CASES_PATH,
    DEFAULT_BRAND_KNOWLEDGE_PATH,
    DEFAULT_BRAND_LIBRARY_PATH,
    DEFAULT_CASE_STORE_PATH,
    DEFAULT_CHROMA_DIRECTORY,
    DEFAULT_EVALUATION_RUNS_PATH,
    DEFAULT_GOLDEN_SET_V4_PATH,
    DEFAULT_POLICY_PATH,
    DEFAULT_PROMPT_PRESETS_PATH,
    DEFAULT_REVIEW_RECORDS_PATH,
    DEFAULT_SKILL_PRESETS_PATH,
    DEFAULT_STAGING_DATASET_PATH,
    DEFAULT_TRACE_PATH,
    DEFAULT_CUSTOM_PROMPTS_PATH,
    DEFAULT_CUSTOM_SKILLS_PATH,
    DEFAULT_MODEL_EXPERIMENTS_PATH,
    DEFAULT_REVIEWER_PROMPTS_PATH,
    DEFAULT_WORKFLOW_RELEASES_PATH,
    DEFAULT_KNOWLEDGE_DOCUMENTS_PATH,
    DEFAULT_SYNTHETIC_DEMO_CASES_PATH,
    PROJECT_ROOT,
)
from trust_safety_agent.bad_cases import BadCaseStore
from trust_safety_agent.brand_knowledge import BrandKnowledgeStore, load_brand_knowledge
from trust_safety_agent.case_store import AgentRunRecord, CaseStore, JsonlStore, ProductCase, RunDecision, ShopCase
from trust_safety_agent.model_experiments import WorkflowConfigurationStore
from trust_safety_agent.phase9_ui import (
    render_bad_cases as render_phase9_bad_cases,
    render_brand_knowledge as render_phase9_brand_knowledge,
    render_candidate_dataset as render_phase9_candidate_dataset,
    render_case_list as render_phase9_case_list,
    render_comprehensive_review as render_phase9_comprehensive_review,
    render_model_release as render_phase9_model_release,
    render_my_prompts as render_phase9_my_prompts,
    render_product_review as render_phase9_product_review,
    render_reviewer_prompt_observation as render_phase9_reviewer_prompt_observation,
    render_shop_review as render_phase9_shop_review,
    render_trace_replay as render_phase9_trace_replay,
    render_workflow_management as render_phase9_workflow_management,
)
from trust_safety_agent.phase9_workflow import ProductWorkflow, ReviewRouter, ShopWorkflow, WorkflowTraceRecorder
from trust_safety_agent.price_search import OfflineFixturePriceProvider
from trust_safety_agent.reviewer_prompts import ExperimentalPromptResult, ReviewerPromptStore, ReviewerPromptVersion
from trust_safety_agent.workflow_releases import WorkflowName, WorkflowReleaseStore
from trust_safety_agent.knowledge_documents import KnowledgeDocumentStore, KnowledgeKind
from trust_safety_agent.controlled_agent import AgentRequest, ControlledPolicyAgent
from trust_safety_agent.controlled_tools import build_default_tool_registry
from trust_safety_agent.evaluation_lab import (
    EvaluationLabStore,
    run_offline_rules_experiment,
)
from trust_safety_agent.llm_adjudicator import LLMPolicyAdjudicator
from trust_safety_agent.llm_client import ImageInput
from trust_safety_agent.resilient_llm_client import (
    ResilientLLMSettings,
    ResilientOpenAICompatibleClient,
)
from trust_safety_agent.policy_loader import load_policy_file, load_policy_text
from trust_safety_agent.product_review import (
    BrandAuthorizationStatus,
    DesignSimilaritySignal,
    ImageQuality,
    LogoMatchType,
    ProductReviewAssistant,
    ProductReviewDecision,
    ProductReviewInput,
    ProductReviewResult,
)
from trust_safety_agent.prompt_skills import (
    PromptSkillRegistry,
    PromptVersion,
    PromptWorkflow,
    SkillDefinition,
)
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    PolicyChunk,
)
from trust_safety_agent.reviewer_workspace import (
    ReviewerFeedback,
    ReviewerLabel,
    ReviewerWorkspaceStore,
    ReviewWorkflow,
)
from trust_safety_agent.shop_identity import (
    AuthorizationStatus,
    ShopIdentityInput,
    ShopIdentityReviewer,
)
from trust_safety_agent.staging_dataset import StagingDatasetStore
from trust_safety_agent.synthetic_dataset import (
    DemoWorkflow,
    load_synthetic_demo_cases,
)
from trust_safety_agent.trace import TraceStore
from trust_safety_agent.trace_builders import (
    controlled_agent_trace,
    product_review_trace,
    shop_identity_trace,
)
from trust_safety_agent.vector_store import PolicyVectorStore


st.set_page_config(
    page_title="Trust & Safety AI 决策实验室",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --lab-blue: #2563eb;
        --lab-ink: #111827;
        --lab-muted: #64748b;
        --lab-line: #e2e8f0;
        --lab-surface: #ffffff;
        --lab-soft: #f8fafc;
        --lab-green: #15803d;
        --lab-red: #b91c1c;
        --lab-amber: #b45309;
    }
    .block-container {max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: 0 !important; color: var(--lab-ink);}
    h1 {font-size: 2rem !important; line-height: 1.2 !important;}
    h2 {font-size: 1.45rem !important;}
    h3 {font-size: 1.16rem !important;}
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {border-right: 1px solid var(--lab-line);}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size: .88rem;}
    [data-testid="stForm"] {border: 0; padding: 0;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--lab-line) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    .lab-brand {padding: .25rem 0 1rem;}
    .lab-brand__name {font-size: 1.22rem; line-height: 1.25; font-weight: 780; color: var(--lab-ink);}
    .lab-brand__meta {font-size: .78rem; color: var(--lab-muted); margin-top: .3rem;}
    .lab-page-header {border-bottom: 1px solid var(--lab-line); padding-bottom: 1rem; margin-bottom: 1.25rem;}
    .lab-eyebrow {font-size: .74rem; font-weight: 700; color: var(--lab-blue); text-transform: uppercase;}
    .lab-page-title {font-size: 1.75rem; line-height: 1.25; font-weight: 760; color: var(--lab-ink); margin-top: .25rem;}
    .lab-page-desc {font-size: .92rem; color: var(--lab-muted); margin-top: .4rem; max-width: 760px;}
    .lab-status-line {display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .75rem;}
    .lab-chip {display: inline-flex; align-items: center; min-height: 26px; padding: 3px 9px;
        border-radius: 999px; border: 1px solid var(--lab-line); background: var(--lab-soft);
        color: #334155; font-size: .75rem; font-weight: 650;}
    .lab-chip--ok {color: var(--lab-green); border-color: #bbf7d0; background: #f0fdf4;}
    .lab-chip--warn {color: var(--lab-amber); border-color: #fde68a; background: #fffbeb;}
    .lab-result {border-left: 4px solid var(--lab-blue); padding: .9rem 1rem; background: var(--lab-soft); margin-bottom: 1rem;}
    .lab-result--reject {border-left-color: var(--lab-red); background: #fef2f2;}
    .lab-result--approve {border-left-color: var(--lab-green); background: #f0fdf4;}
    .lab-result--review {border-left-color: var(--lab-amber); background: #fffbeb;}
    .lab-result__label {font-size: .74rem; color: var(--lab-muted); font-weight: 700;}
    .lab-result__value {font-size: 1.45rem; color: var(--lab-ink); font-weight: 780; margin-top: .1rem;}
    .lab-empty {min-height: 260px; display: flex; flex-direction: column; justify-content: center;
        border: 1px dashed #cbd5e1; padding: 1.5rem; color: var(--lab-muted); background: var(--lab-soft);}
    .lab-empty strong {color: var(--lab-ink); font-size: 1rem; margin-bottom: .4rem;}
    .lab-step {display: flex; gap: .7rem; padding: .65rem 0; border-bottom: 1px solid var(--lab-line);}
    .lab-step:last-child {border-bottom: 0;}
    .lab-step__index {width: 24px; height: 24px; border-radius: 50%; display: inline-flex;
        align-items: center; justify-content: center; background: #dbeafe; color: #1d4ed8;
        font-size: .72rem; font-weight: 750; flex: 0 0 24px;}
    .lab-step__title {font-size: .86rem; font-weight: 680; color: var(--lab-ink);}
    .lab-step__meta {font-size: .76rem; color: var(--lab-muted); margin-top: .15rem;}
    @media (max-width: 640px) {
        .block-container {padding-top: 1rem;}
        .lab-page-title {font-size: 1.45rem;}
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--lab-line);
        border-radius: 8px;
        padding: 10px 14px;
        background: var(--lab-surface);
    }
    [data-testid="stMetricLabel"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store() -> PolicyVectorStore:
    return PolicyVectorStore(DEFAULT_CHROMA_DIRECTORY)


@st.cache_resource
def get_brand_library() -> ControlledBrandLibrary:
    return ControlledBrandLibrary.from_csv_and_knowledge(
        DEFAULT_BRAND_LIBRARY_PATH,
        DEFAULT_BRAND_KNOWLEDGE_PATH,
    )


@st.cache_resource
def get_brand_store() -> BrandKnowledgeStore:
    store = BrandKnowledgeStore(DEFAULT_CHROMA_DIRECTORY)
    if (
        store.count() != 50
        or store.collection.metadata.get("embedding_model") != store.embedder.name
    ):
        store.index(load_brand_knowledge(DEFAULT_BRAND_KNOWLEDGE_PATH))
    return store


@st.cache_resource
def get_case_store() -> CaseStore:
    return CaseStore(DEFAULT_CASE_STORE_PATH)


@st.cache_resource
def get_phase9_run_store() -> JsonlStore:
    return JsonlStore(DEFAULT_AGENT_RUNS_PATH, AgentRunRecord)


@st.cache_resource
def get_reviewer_prompt_store() -> ReviewerPromptStore:
    return ReviewerPromptStore(DEFAULT_REVIEWER_PROMPTS_PATH)


@st.cache_resource
def get_workflow_release_store() -> WorkflowReleaseStore:
    return WorkflowReleaseStore(DEFAULT_WORKFLOW_RELEASES_PATH)


@st.cache_resource
def get_knowledge_document_store() -> KnowledgeDocumentStore:
    return KnowledgeDocumentStore(DEFAULT_KNOWLEDGE_DOCUMENTS_PATH)


@st.cache_resource
def get_bad_case_store() -> BadCaseStore:
    return BadCaseStore(DEFAULT_BAD_CASES_PATH)


@st.cache_resource
def get_model_configuration_store() -> WorkflowConfigurationStore:
    return WorkflowConfigurationStore(DEFAULT_MODEL_EXPERIMENTS_PATH)


@st.cache_resource
def get_review_store() -> ReviewerWorkspaceStore:
    return ReviewerWorkspaceStore(DEFAULT_REVIEW_RECORDS_PATH)


@st.cache_resource
def get_staging_store() -> StagingDatasetStore:
    return StagingDatasetStore(DEFAULT_STAGING_DATASET_PATH)


@st.cache_resource
def get_trace_store() -> TraceStore:
    return TraceStore(DEFAULT_TRACE_PATH)


def get_prompt_skill_registry() -> PromptSkillRegistry:
    return PromptSkillRegistry(
        DEFAULT_PROMPT_PRESETS_PATH,
        DEFAULT_SKILL_PRESETS_PATH,
        DEFAULT_CUSTOM_PROMPTS_PATH,
        DEFAULT_CUSTOM_SKILLS_PATH,
    )


def ensure_default_index(store: PolicyVectorStore) -> None:
    chunks = store.list_chunks()
    if (
        chunks
        and store.collection.metadata.get("embedding_model") != store.embedder.name
    ):
        store.index(chunks, replace=True)
        chunks = store.list_chunks()
    expected_source = DEFAULT_POLICY_PATH.name
    default_is_present = any(
        chunk.source == expected_source and chunk.policy_version == "v2.0.0"
        for chunk in chunks
    )
    if not chunks:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))
    elif not default_is_present:
        store.index(load_policy_file(DEFAULT_POLICY_PATH), replace=False)


def chunk_rows(chunks: List[PolicyChunk]) -> List[dict]:
    return [
        {
            "策略 ID": chunk.policy_id,
            "标题": chunk.title,
            "策略标签": chunk.policy_label.value if chunk.policy_label else "",
            "豁免类型": (
                "" if chunk.exemption_type.value == "none" else chunk.exemption_type.value
            ),
            "分块 ID": chunk.chunk_id,
            "字符数": len(chunk.content),
            "内容预览": chunk.content[:180].replace("\n", " "),
        }
        for chunk in chunks
    ]


def runtime_llm_settings() -> ResilientLLMSettings:
    env_settings = ResilientLLMSettings.from_env()
    return ResilientLLMSettings(
        api_key=st.session_state.get("llm_api_key", env_settings.api_key),
        api_base=st.session_state.get("llm_api_base", env_settings.api_base),
        model=st.session_state.get("llm_model", env_settings.model),
        timeout_seconds=env_settings.timeout_seconds,
        response_format=st.session_state.get(
            "llm_response_format", env_settings.response_format
        ),
        fallback_models=tuple(
            item.strip()
            for item in st.session_state.get(
                "llm_fallback_models", ", ".join(env_settings.fallback_models)
            ).split(",")
            if item.strip()
        ),
    )


def build_reviewer_prompt_runner(
    settings: ResilientLLMSettings,
    policy_store: PolicyVectorStore,
    brand_store: BrandKnowledgeStore,
    trace_store: TraceStore,
    run_store: JsonlStore,
):
    if not settings.configured:
        return None

    def run(case: ProductCase | ShopCase, prompt: ReviewerPromptVersion):
        if isinstance(case, ProductCase):
            content_type = ContentType.PRODUCT
            text = (
                f"product_id={case.product_id}\ntitle={case.title}\n"
                f"description={case.description}\nbrand={case.brand}\n"
                f"brand_authorized={case.brand_authorized}\nprice={case.price} {case.currency or ''}"
            )
            media = case.product_images
        else:
            content_type = ContentType.SHOP_AVATAR
            text = (
                f"shop_id={case.shop_id}\nshop_name={case.shop_name}\n"
                f"brand={case.brand}\nbrand_authorized={case.brand_authorized}"
            )
            media = [case.shop_avatar]

        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        case_id = case.product_id if isinstance(case, ProductCase) else case.shop_id
        workflow = "product_prompt_experiment" if isinstance(case, ProductCase) else "shop_prompt_experiment"
        prompt_ref = f"{prompt.prompt_id}@v{prompt.version}"
        recorder = WorkflowTraceRecorder(
            run_id=run_id,
            case_id=case_id,
            case_revision=case.revision,
            workflow=workflow,
            prompt_version=prompt_ref,
            skill_version="not_used",
            model_id=settings.model,
        )
        recorder.execute(
            "validate_input",
            {"case_id": case_id, "workflow": workflow},
            lambda: {"valid": bool(text.strip() and media), "image_count": len(media)},
        )

        instructions = prompt.instructions
        policy_hits = policy_store.retrieve(text, top_k=6) if prompt.use_policy_rag else []
        recorder.execute(
            "retrieve_policy",
            {"enabled": prompt.use_policy_rag, "query": text[:500]},
            lambda: {"selected_chunk_ids": [hit.chunk.chunk_id for hit in policy_hits[:3]]},
            candidate_evidence=[{"chunk_id": hit.chunk.chunk_id, "score": hit.score} for hit in policy_hits],
        )
        if prompt.use_brand_rag:
            hits = brand_store.retrieve(text, top_k=5)
            context = "\n".join(
                f"- {hit.chunk.brand_name}: aliases={', '.join(hit.chunk.aliases)}; "
                f"category={hit.chunk.category}; products={', '.join(hit.chunk.representative_products)}"
                for hit in hits
            )
            instructions += f"\n\nBRAND KNOWLEDGE CONTEXT\n{context}"
        else:
            hits = []
        recorder.execute(
            "retrieve_brand_knowledge",
            {"enabled": prompt.use_brand_rag, "query": text[:500]},
            lambda: {"selected_chunk_ids": [hit.chunk.chunk_id for hit in hits[:3]]},
            candidate_evidence=[{"chunk_id": hit.chunk.chunk_id, "score": hit.score} for hit in hits],
        )

        images = []
        for reference in media:
            if reference.startswith(("http://", "https://")):
                images.append(ImageInput(url=reference))
                continue
            path = PROJECT_ROOT / reference
            if path.exists() and path.is_file():
                media_type = "image/png" if path.suffix.casefold() == ".png" else "image/jpeg"
                images.append(ImageInput(data=path.read_bytes(), media_type=media_type))

        adjudicator = LLMPolicyAdjudicator(
            policy_store,
            ResilientOpenAICompatibleClient(settings),
            additional_instructions=instructions,
            use_policy_rag=prompt.use_policy_rag,
        )
        output = recorder.execute(
            "model_inference",
            {"prompt_ref": prompt_ref, "model_id": settings.model, "image_count": len(images)},
            lambda: adjudicator.adjudicate(case_id, content_type, text, images).model_dump(mode="json"),
        )
        decision = AgentDecision.model_validate(output)
        result = ExperimentalPromptResult(
            case_id=case_id,
            prompt_ref=prompt_ref,
            decision=decision.decision.value,
            confidence=decision.confidence,
            reason=decision.reason,
            recommended_action=decision.recommended_action,
            trace_id=recorder.trace_id,
        )
        recorder.execute("publish_experimental_result", {"case_id": case_id}, lambda: result.model_dump(mode="json"))
        trace_store.append(recorder.finish(result.model_dump(mode="json")))
        run_decision = {
            AgentDecisionLabel.APPROVE: RunDecision.APPROVE,
            AgentDecisionLabel.REJECT: RunDecision.REJECT,
            AgentDecisionLabel.NEED_REVIEW: RunDecision.MANUAL_REVIEW,
        }[decision.decision]
        run_store.append(AgentRunRecord(
            run_id=run_id,
            case_id=case_id,
            case_revision=case.revision,
            workflow_version="v9.1.0-experimental",
            prompt_version=prompt_ref,
            skill_version="not_used",
            model_id=settings.model,
            policy_index_version="synthetic-policy-v2" if prompt.use_policy_rag else "disabled",
            brand_index_version="v1.0.0" if prompt.use_brand_rag else "disabled",
            suggested_decision=run_decision,
            confidence=decision.confidence,
            trace_id=recorder.trace_id,
            experimental=True,
        ))
        return result

    return run


def render_page_header(
    eyebrow: str,
    title: str,
    description: str,
    chips: List[tuple[str, str]] | None = None,
) -> None:
    chip_html = "".join(
        f'<span class="lab-chip {escape(style)}">{escape(label)}</span>'
        for label, style in (chips or [])
    )
    st.markdown(
        f"""
        <div class="lab-page-header">
            <div class="lab-eyebrow">{escape(eyebrow)}</div>
            <div class="lab-page-title">{escape(title)}</div>
            <div class="lab-page-desc">{escape(description)}</div>
            <div class="lab-status-line">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_settings(store: PolicyVectorStore) -> None:
    render_page_header(
        "资产与设置",
        "系统设置",
        "集中管理模型连接和政策索引。面试演示时无需进入此页面。",
    )
    env_settings = ResilientLLMSettings.from_env()
    model_col, index_col = st.columns(2, gap="large")
    with model_col, st.container(border=True):
        st.subheader("模型连接")
        api_base = st.text_input(
            "API 地址",
            value=st.session_state.get("llm_api_base", env_settings.api_base),
            key="llm_api_base",
        )
        model = st.text_input(
            "视觉模型",
            value=env_settings.model,
            key="llm_model",
        )
        fallback_models = st.text_input(
            "备用模型（逗号分隔）",
            value=", ".join(env_settings.fallback_models),
            key="llm_fallback_models",
        )
        response_format = st.selectbox(
            "结构化输出模式",
            options=["auto", "json_schema", "json_object"],
            index=["auto", "json_schema", "json_object"].index(
                env_settings.response_format
            ),
            key="llm_response_format",
        )
        api_key = st.text_input(
            "API Key",
            value=env_settings.api_key,
            type="password",
            key="llm_api_key",
        )
        if api_key.strip():
            st.success("模型连接参数已配置。")
        else:
            st.warning("需要 API Key 才能运行模型与 Agent。")

    with index_col, st.container(border=True):
        st.subheader("策略索引")
        uploaded = st.file_uploader("Markdown 策略文档", type=["md", "txt"])
        max_chars = st.number_input(
            "分块大小",
            min_value=300,
            max_value=2000,
            value=900,
            step=100,
        )
        overlap_chars = st.number_input(
            "重叠字符数",
            min_value=0,
            max_value=400,
            value=120,
            step=20,
        )

        if st.button(
            ":material/database: 构建索引",
            type="primary",
            width="stretch",
        ):
            if overlap_chars >= max_chars:
                st.error("重叠字符数必须小于分块大小。")
            else:
                if uploaded:
                    markdown = uploaded.getvalue().decode("utf-8")
                    chunks = load_policy_text(
                        markdown,
                        source=uploaded.name[:500],
                        max_chars=int(max_chars),
                        overlap_chars=int(overlap_chars),
                    )
                else:
                    chunks = load_policy_file(
                        DEFAULT_POLICY_PATH,
                        max_chars=int(max_chars),
                        overlap_chars=int(overlap_chars),
                    )
                store.index(chunks)
                st.toast(f"已建立 {len(chunks)} 个策略分块的索引。")
                st.rerun()

        st.caption(f"嵌入模型：`{store.embedder.name}`")
        st.caption(f"向量集合：`{store.collection_name}`")


def render_search(store: PolicyVectorStore) -> None:
    st.subheader("策略检索")
    query_col, k_col = st.columns([5, 1])
    with query_col:
        query = st.text_input(
            "案例或策略查询",
            value="Gucci replica handbag sold as a 1:1 copy",
        )
    with k_col:
        top_k = st.number_input("返回数量 Top K", min_value=1, max_value=10, value=3)

    if query.strip():
        for hit in store.retrieve(query, top_k=int(top_k)):
            label = (
                hit.chunk.policy_label.value
                if hit.chunk.policy_label
                else hit.chunk.exemption_type.value
            )
            with st.expander(
                f"#{hit.rank} · {hit.chunk.policy_id} · {hit.chunk.title} "
                f"· 相关度 {hit.score:.3f}",
                expanded=hit.rank == 1,
            ):
                st.caption(
                    f"`{label}` · `{hit.chunk.chunk_id}` · "
                    f"{' > '.join(hit.chunk.heading_path)}"
                )
                st.markdown(hit.chunk.content)


def render_decision(
    decision: AgentDecision,
    engine: str,
    image_used: bool,
) -> None:
    status = {
        AgentDecisionLabel.REJECT: "拒绝",
        AgentDecisionLabel.APPROVE: "通过",
        AgentDecisionLabel.NEED_REVIEW: "人工复核",
    }[decision.decision]
    style = {
        AgentDecisionLabel.REJECT: "reject",
        AgentDecisionLabel.APPROVE: "approve",
        AgentDecisionLabel.NEED_REVIEW: "review",
    }[decision.decision]
    st.markdown(
        f"""
        <div class="lab-result lab-result--{style}">
            <div class="lab-result__label">最终决策</div>
            <div class="lab-result__value">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    policy_labels = {
        "counterfeit": "假货",
        "knockoff": "山寨仿款",
        "trademark_misuse": "商标误用",
        "shop_impersonation": "店铺冒充",
        "risky_query": "风险查询",
    }
    metric_row_one = st.columns(2)
    metric_row_one[0].metric("置信度", f"{decision.confidence:.0%}")
    metric_row_one[1].metric(
        "策略类型",
        policy_labels.get(
            decision.policy_label.value if decision.policy_label else "", "—"
        ),
    )
    metric_row_two = st.columns(2)
    metric_row_two[0].metric("证据数量", len(decision.matched_policy))
    metric_row_two[1].metric("图像证据", "已使用" if image_used else "未使用")

    st.caption(f"推理引擎：`{engine}`")

    display_text = {
        "The content explicitly promotes imitation of a named branded design.": "内容明确宣传对受保护品牌设计的仿制。",
        "Explicit counterfeit terminology appears with a protected brand.": "受控品牌与明确的假货表述同时出现。",
        "A protected trademark is used to promote or identify an unrelated item.": "受保护商标被用于宣传或标识无关商品。",
        "The content presents a seller as an official or authorized brand shop.": "内容将卖家呈现为官方或授权品牌店铺。",
        "No violation signal or protected brand reference was detected.": "未识别到违规信号或受保护品牌指向。",
        "Reject the content and record the cited policy.": "拒绝该内容，并记录引用的政策证据。",
        "Approve the content.": "通过该内容。",
        "Send the case to a human reviewer.": "将案例转交人工复核。",
        "Review context, authorization, and media evidence.": "进一步核查上下文、授权和媒体证据。",
    }
    st.markdown("**判断理由**")
    st.write(display_text.get(decision.reason, decision.reason))
    st.markdown("**建议操作**")
    st.write(
        display_text.get(decision.recommended_action, decision.recommended_action)
    )

    if decision.matched_policy:
        st.markdown("**策略证据**")
        for evidence in decision.matched_policy:
            with st.expander(
                f"{evidence.policy_id} · {evidence.chunk_id}",
                expanded=True,
            ):
                if evidence.retrieval_score is not None:
                    st.caption(f"检索相关度：{evidence.retrieval_score:.3f}")
                st.markdown(f"> {evidence.quote}")

    with st.expander("结构化输出"):
        st.code(decision.model_dump_json(indent=2), language="json")


DEMO_CASES: Dict[str, Dict[str, Any]] = {
    "山寨仿款": {
        "case_id": "DEMO-KO-001",
        "content_type": ContentType.PRODUCT,
        "title": "Gucci 同款手提包",
        "description": "Gucci dupe，same design as the original，非专柜正品。",
        "brand": "Gucci",
        "ocr_text": "Gucci",
        "task": "判断商品是否属于山寨仿款，并引用有效政策证据。",
    },
    "明确假货": {
        "case_id": "DEMO-CF-001",
        "content_type": ContentType.PRODUCT,
        "title": "Gucci 1:1 镜像手提包",
        "description": "Replica mirror copy，附品牌包装，明确非正品。",
        "brand": "Gucci",
        "ocr_text": "Gucci 1:1",
        "task": "完成商品知识产权审核，明确区分假货与山寨政策。",
    },
    "二手豁免": {
        "case_id": "DEMO-EX-001",
        "content_type": ContentType.PRODUCT,
        "title": "个人闲置 Gucci 二手手袋",
        "description": "自用闲置转卖，提供购买凭证和实物磨损照片，不宣称新品。",
        "brand": "Gucci",
        "ocr_text": "",
        "task": "检查二手交易豁免是否适用，证据不足时转人工。",
    },
    "证据不足": {
        "case_id": "DEMO-MR-001",
        "content_type": ContentType.PRODUCT,
        "title": "Luxury style handbag",
        "description": "商品图片模糊，未提供品牌授权或可识别的设计细节。",
        "brand": "",
        "ocr_text": "",
        "task": "在证据不足时避免过度判断，并给出人工复核建议。",
    },
}


for synthetic_case in load_synthetic_demo_cases(DEFAULT_SYNTHETIC_DEMO_CASES_PATH):
    if synthetic_case.workflow != DemoWorkflow.PRODUCT_IPR:
        continue
    DEMO_CASES[f"V5 · {synthetic_case.name}"] = {
        "case_id": synthetic_case.case_id,
        "content_type": ContentType.PRODUCT,
        "title": synthetic_case.title,
        "description": synthetic_case.description,
        "brand": synthetic_case.optional_brand_field or "",
        "ocr_text": synthetic_case.ocr_text,
        "task": "依据合成政策 v2 完成多模态商品知识产权审核，证据不足时转人工。",
        "image_path": synthetic_case.image_path,
        "visual_marks": synthetic_case.visual_marks,
        "brand_authorization_status": synthetic_case.brand_authorization_status,
        "observed_product_brand": synthetic_case.observed_product_brand or "",
        "logo_match_type": synthetic_case.logo_match_type,
        "design_similarity": synthetic_case.design_similarity,
        "independent_brand_registered": synthetic_case.independent_brand_registered,
        "listing_price": synthetic_case.listing_price,
        "reference_price": synthetic_case.reference_price,
        "image_quality": synthetic_case.image_quality,
        "visual_confidence": synthetic_case.visual_confidence,
        "expected_decision": synthetic_case.expected_decision,
        "expected_policy_id": synthetic_case.expected_policy_id,
    }


def reset_workbench_result() -> None:
    for key in (
        "workbench_result",
        "workbench_result_kind",
        "workbench_engine",
        "workbench_image_used",
        "workbench_skill",
        "workbench_mode_used",
        "workbench_elapsed_ms",
    ):
        st.session_state.pop(key, None)


def render_execution_steps(mode: str, result: Any) -> None:
    if mode == "受控 Agent":
        executions = list(getattr(result, "tool_executions", []))
        steps = [("案例解析", "已读取任务、案例字段与 Skill 边界")]
        steps.extend(
            (
                f"调用 {item.tool_name}",
                f"{item.status.value} · {item.latency_ms} ms",
            )
            for item in executions
        )
        steps.append(("决策门禁", f"停止原因：{result.stop_reason.value}"))
    elif mode == "多模态模型":
        steps = [
            ("案例解析", "合并文本与图像输入"),
            ("政策检索", f"返回 {len(result.matched_policy)} 条有效证据"),
            ("模型推理", "执行版本化 Prompt 与结构化输出"),
            ("本地门禁", "校验政策、豁免、置信度和证据"),
        ]
    else:
        evidence_count = (
            len(result.policy_references)
            if isinstance(result, ProductReviewResult)
            else len(result.matched_policy)
        )
        steps = [
            ("案例解析", "标准化内容类型与文本字段"),
            ("规则识别", "匹配授权、Logo、设计与信息层关系"),
            ("政策检索", f"返回 {evidence_count} 条有效证据"),
            ("结果输出", "使用离线基线形成审核建议"),
        ]
    html = "".join(
        f"""
        <div class="lab-step">
            <span class="lab-step__index">{index}</span>
            <div><div class="lab-step__title">{escape(title)}</div>
            <div class="lab-step__meta">{escape(meta)}</div></div>
        </div>
        """
        for index, (title, meta) in enumerate(steps, start=1)
    )
    st.markdown(html, unsafe_allow_html=True)


def render_agent_result(result: Any) -> None:
    label = {
        ProductReviewDecision.REJECT: "拒绝",
        ProductReviewDecision.APPROVE: "通过",
        ProductReviewDecision.MANUAL_REVIEW: "人工复核",
    }[result.decision]
    style = {
        ProductReviewDecision.REJECT: "reject",
        ProductReviewDecision.APPROVE: "approve",
        ProductReviewDecision.MANUAL_REVIEW: "review",
    }[result.decision]
    st.markdown(
        f"""
        <div class="lab-result lab-result--{style}">
            <div class="lab-result__label">Agent 最终建议</div>
            <div class="lab-result__value">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    row_one = st.columns(2)
    row_one[0].metric("置信度", result.confidence.value)
    row_one[1].metric("政策 ID", result.policy_id or "—")
    row_two = st.columns(2)
    row_two[0].metric("工具调用", len(result.tool_executions))
    row_two[1].metric("Token", result.total_tokens)
    raw_reason = str(result.reason)
    lowered_reason = raw_reason.lower()
    if "llmclienterror" in lowered_reason or "connection failed" in lowered_reason:
        display_reason = "模型服务暂时不可用，系统已按安全策略转交人工复核。"
    elif "validation" in lowered_reason or "schema" in lowered_reason:
        display_reason = "模型输出未通过结构化校验，系统已转交人工复核。"
    elif "duplicate" in lowered_reason:
        display_reason = "检测到重复工具调用，系统已停止自动执行并转交人工复核。"
    else:
        display_reason = raw_reason
    st.markdown("**判断理由**")
    st.write(display_reason)
    if display_reason != raw_reason:
        with st.expander("技术详情"):
            st.code(raw_reason, language="text")


def render_product_result(result: ProductReviewResult) -> None:
    label = {
        ProductReviewDecision.REJECT: "拒绝",
        ProductReviewDecision.APPROVE: "通过",
        ProductReviewDecision.MANUAL_REVIEW: "人工复核",
    }[result.suggested_decision]
    style = {
        ProductReviewDecision.REJECT: "reject",
        ProductReviewDecision.APPROVE: "approve",
        ProductReviewDecision.MANUAL_REVIEW: "review",
    }[result.suggested_decision]
    st.markdown(
        f"""
        <div class="lab-result lab-result--{style}">
            <div class="lab-result__label">结构化商品审核建议</div>
            <div class="lab-result__value">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    row_one = st.columns(2)
    row_one[0].metric("置信度", result.confidence.value)
    row_one[1].metric(
        "风险子类型",
        result.risk_subtype.value if result.risk_subtype else "—",
    )
    row_two = st.columns(2)
    row_two[0].metric("策略证据", len(result.policy_references))
    row_two[1].metric("品牌候选", len(result.candidate_brands))
    st.markdown("**判断理由**")
    st.write(result.reason)
    if result.policy_references:
        st.markdown("**策略证据**")
        for reference in result.policy_references:
            with st.expander(
                f"{reference.policy_id} · {reference.title}", expanded=True
            ):
                st.caption(f"检索相关度：{reference.retrieval_score:.3f}")
                st.markdown(reference.quote)
    if result.reviewer_checkpoints:
        st.markdown("**人工复核要点**")
        for checkpoint in result.reviewer_checkpoints:
            st.markdown(f"- {checkpoint}")
    with st.expander("结构化输出"):
        st.code(result.model_dump_json(indent=2), language="json")


def render_review_workbench(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
    trace_store: TraceStore,
    llm_settings: ResilientLLMSettings,
) -> None:
    render_page_header(
        "决策工作台",
        "AI 审核工作台",
        "选择案例和运行模式，在同一页面完成输入、工具执行、政策证据与最终决策。",
        [
            ("模型已配置" if llm_settings.configured else "模型未配置", "lab-chip--ok" if llm_settings.configured else "lab-chip--warn"),
            (f"政策 {len({item.policy_id for item in store.list_chunks()})} 组", ""),
            ("证据不足自动转人工", ""),
        ],
    )
    default_mode = "受控 Agent" if llm_settings.configured else "离线基线"
    if st.session_state.get("workbench_mode") not in {
        "受控 Agent",
        "多模态模型",
        "离线基线",
    }:
        st.session_state["workbench_mode"] = default_mode
    if st.session_state.get("workbench_preset") not in DEMO_CASES:
        st.session_state["workbench_preset"] = "山寨仿款"

    preset_col, mode_col = st.columns([1, 1.35], gap="large")
    with preset_col:
        preset_name = st.selectbox(
            "演示案例",
            options=list(DEMO_CASES),
            help="切换案例会加载一套可直接运行的演示数据。",
            key="workbench_preset",
            on_change=reset_workbench_result,
        )
    with mode_col:
        mode = st.segmented_control(
            "运行模式",
            options=["受控 Agent", "多模态模型", "离线基线"],
            key="workbench_mode",
            on_change=reset_workbench_result,
        )
    preset = DEMO_CASES[preset_name]
    widget_scope = f"{preset_name}-{mode}"
    registry = get_prompt_skill_registry()
    workflow = (
        PromptWorkflow.CONTROLLED_AGENT
        if mode == "受控 Agent"
        else PromptWorkflow.GENERAL_CASE
    )
    skills = sorted(
        registry.skills(workflow),
        key=lambda item: tuple(int(value) for value in item.version[1:].split(".")),
        reverse=True,
    )
    built_in_image_path = (
        PROJECT_ROOT / preset["image_path"] if preset.get("image_path") else None
    )

    input_col, result_col = st.columns([0.92, 1.08], gap="large")
    with input_col, st.container(border=True):
        st.subheader("案例输入")
        with st.form(f"unified-review-{preset_name}-{mode}"):
            selected_skill = None
            resolved = None
            if mode != "离线基线":
                selected_skill = st.selectbox(
                    "审核 Skill",
                    options=skills,
                    format_func=lambda item: f"{item.name} · {item.version}",
                    key=f"workbench-skill-{widget_scope}",
                )
                resolved = registry.resolve_skill(
                    selected_skill.skill_id, selected_skill.version
                )
            id_col, type_col = st.columns(2)
            case_id = id_col.text_input(
                "案例 ID",
                value=preset["case_id"],
                key=f"workbench-case-id-{widget_scope}",
            )
            content_type = type_col.selectbox(
                "内容类型",
                options=list(ContentType),
                index=list(ContentType).index(preset["content_type"]),
                format_func=lambda value: {
                    ContentType.PRODUCT: "商品",
                    ContentType.VIDEO: "视频",
                    ContentType.SHOP_NAME: "店铺名称",
                    ContentType.SHOP_AVATAR: "店铺头像",
                    ContentType.QUERY: "搜索查询",
                }[value],
                key=f"workbench-content-type-{widget_scope}",
            )
            title = st.text_input(
                "标题",
                value=preset["title"],
                key=f"workbench-title-{widget_scope}",
            )
            description = st.text_area(
                "描述",
                value=preset["description"],
                height=105,
                max_chars=2000,
                key=f"workbench-description-{widget_scope}",
            )
            brand_col, ocr_col = st.columns(2)
            brand = brand_col.text_input(
                "品牌字段",
                value=preset["brand"],
                key=f"workbench-brand-{widget_scope}",
            )
            ocr_text = ocr_col.text_input(
                "OCR 文本",
                value=preset["ocr_text"],
                key=f"workbench-ocr-{widget_scope}",
            )
            if built_in_image_path:
                st.image(
                    str(built_in_image_path),
                    caption="合成演示图片，不包含真实品牌或公司素材",
                    width="stretch",
                )
            image_col, url_col = st.columns(2)
            uploaded_image = image_col.file_uploader(
                "上传图片",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"workbench-image-{preset_name}-{mode}",
            )
            image_url = url_col.text_input(
                "图片 URL",
                placeholder="https://...",
                key=f"workbench-image-url-{widget_scope}",
            )
            with st.expander("高级设置"):
                visual_marks = st.text_area(
                    "视觉证据（每行一项）",
                    value="\n".join(preset.get("visual_marks", [])),
                    height=90,
                    key=f"workbench-visual-marks-{widget_scope}",
                )
                observed_product_brand = st.text_input(
                    "图片中观察到的商品品牌",
                    value=preset.get("observed_product_brand", ""),
                    key=f"workbench-observed-brand-{widget_scope}",
                )
                signal_col_one, signal_col_two = st.columns(2)
                brand_authorization_status = signal_col_one.selectbox(
                    "品牌授权状态",
                    options=list(BrandAuthorizationStatus),
                    index=list(BrandAuthorizationStatus).index(
                        preset.get(
                            "brand_authorization_status",
                            BrandAuthorizationStatus.UNKNOWN,
                        )
                    ),
                    format_func=lambda value: {
                        BrandAuthorizationStatus.AUTHORIZED: "已授权",
                        BrandAuthorizationStatus.UNAUTHORIZED: "未授权",
                        BrandAuthorizationStatus.UNKNOWN: "未知",
                    }[value],
                    key=f"workbench-authorization-{widget_scope}",
                )
                logo_match_type = signal_col_two.selectbox(
                    "Logo 匹配",
                    options=list(LogoMatchType),
                    index=list(LogoMatchType).index(
                        preset.get("logo_match_type", LogoMatchType.NOT_ASSESSED)
                    ),
                    key=f"workbench-logo-match-{widget_scope}",
                )
                design_similarity = signal_col_one.selectbox(
                    "外观相似度信号",
                    options=list(DesignSimilaritySignal),
                    index=list(DesignSimilaritySignal).index(
                        preset.get(
                            "design_similarity",
                            DesignSimilaritySignal.NOT_ASSESSED,
                        )
                    ),
                    key=f"workbench-design-similarity-{widget_scope}",
                )
                image_quality = signal_col_two.selectbox(
                    "图片质量",
                    options=list(ImageQuality),
                    index=list(ImageQuality).index(
                        preset.get("image_quality", ImageQuality.NOT_ASSESSED)
                    ),
                    key=f"workbench-image-quality-{widget_scope}",
                )
                visual_confidence = st.slider(
                    "视觉证据置信度",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(preset.get("visual_confidence", 1.0)),
                    step=0.05,
                    key=f"workbench-visual-confidence-{widget_scope}",
                )
                independent_brand_registered = st.checkbox(
                    "已提供独立品牌注册证据",
                    value=bool(preset.get("independent_brand_registered", False)),
                    key=f"workbench-independent-brand-{widget_scope}",
                )
                price_col_one, price_col_two = st.columns(2)
                listing_price = price_col_one.number_input(
                    "商品价格（0 表示未提供）",
                    min_value=0.0,
                    value=float(preset.get("listing_price") or 0.0),
                    key=f"workbench-listing-price-{widget_scope}",
                )
                reference_price = price_col_two.number_input(
                    "参考价格（0 表示未提供）",
                    min_value=0.0,
                    value=float(preset.get("reference_price") or 0.0),
                    key=f"workbench-reference-price-{widget_scope}",
                )
                task = st.text_area(
                    "Agent 任务",
                    value=preset["task"],
                    height=80,
                    key=f"workbench-task-{widget_scope}",
                )
                customize_prompt = st.checkbox(
                    "临时调整本次 Prompt",
                    disabled=mode == "离线基线",
                    key=f"workbench-custom-prompt-{widget_scope}",
                )
                default_prompt = resolved.prompt.instructions if resolved else ""
                prompt_instructions = st.text_area(
                    "Prompt 指令",
                    value=default_prompt,
                    height=150,
                    disabled=mode == "离线基线" or not customize_prompt,
                    key=f"workbench-prompt-{widget_scope}",
                )
            submitted = st.form_submit_button(
                ":material/play_arrow: 开始审核",
                type="primary",
                width="stretch",
            )

    if submitted:
        try:
            if uploaded_image and image_url.strip():
                raise ValueError("请只选择一种图片来源。")
            image_path = None
            if uploaded_image:
                image_path = f"uploads/{Path(uploaded_image.name).name}"
            elif built_in_image_path:
                image_path = preset["image_path"]
            marks = [value.strip() for value in visual_marks.splitlines() if value.strip()]
            product_item = ProductReviewInput(
                case_id=case_id,
                title=title,
                description=description,
                optional_brand_field=brand or None,
                image_url=image_url.strip() or None,
                image_path=image_path if not image_url.strip() else None,
                ocr_text=ocr_text,
                visual_marks=marks,
                brand_authorization_status=brand_authorization_status,
                observed_product_brand=observed_product_brand or None,
                logo_match_type=logo_match_type,
                design_similarity=design_similarity,
                independent_brand_registered=independent_brand_registered,
                listing_price=listing_price or None,
                reference_price=reference_price or None,
                image_quality=image_quality,
                visual_confidence=visual_confidence,
            )
            input_text = "\n".join(
                [
                    f"title: {title}",
                    f"description: {description}",
                    f"declared_brand: {brand or 'none'}",
                    f"observed_product_brand: {observed_product_brand or 'unknown'}",
                    f"ocr_text: {ocr_text or 'none'}",
                    f"visual_marks: {', '.join(marks) or 'none'}",
                    f"authorization: {brand_authorization_status.value}",
                    f"logo_match_type: {logo_match_type.value}",
                    f"design_similarity: {design_similarity.value}",
                    f"image_quality: {image_quality.value}",
                    f"visual_confidence: {visual_confidence:.2f}",
                    f"listing_price: {listing_price or 'not_provided'}",
                    f"reference_price: {reference_price or 'not_provided'}",
                ]
            )
            started = time.perf_counter()
            with result_col, st.status("正在执行审核链路...", expanded=True) as status:
                st.write("正在解析案例与运行配置")
                if mode == "受控 Agent":
                    if not llm_settings.configured:
                        raise ValueError("请先在系统设置中配置 API Key 和模型。")
                    assert resolved is not None
                    case_data = product_item.model_dump(mode="json")
                    tool_registry = build_default_tool_registry(
                        store, library, review_store
                    )
                    planner = ResilientOpenAICompatibleClient(llm_settings)
                    request = AgentRequest(
                        case_id=case_id, task=task, case_data=case_data
                    )
                    st.write(f"正在运行 {resolved.skill.name}")
                    result = ControlledPolicyAgent(
                        planner,
                        tool_registry,
                        max_steps=resolved.skill.max_steps,
                        allowed_tools=resolved.skill.allowed_tools,
                        additional_instructions=prompt_instructions,
                        policy_scope=resolved.skill.policy_scope,
                    ).run(request)
                    trace_store.append(controlled_agent_trace(request, result))
                    st.session_state["workbench_result_kind"] = "agent"
                    st.session_state["workbench_result"] = result
                    st.session_state["workbench_case"] = case_data
                    st.session_state["workbench_skill"] = (
                        f"{resolved.skill.skill_id}@{resolved.skill.version}"
                    )
                else:
                    images = []
                    if uploaded_image:
                        images.append(
                            ImageInput(
                                data=uploaded_image.getvalue(),
                                media_type=uploaded_image.type or "image/jpeg",
                            )
                        )
                    elif image_url.strip():
                        images.append(ImageInput(url=image_url.strip()))
                    elif built_in_image_path:
                        images.append(
                            ImageInput(
                                data=built_in_image_path.read_bytes(),
                                media_type="image/png",
                            )
                        )
                    if mode == "多模态模型":
                        if not llm_settings.configured:
                            raise ValueError("请先在系统设置中配置 API Key 和模型。")
                        client = ResilientOpenAICompatibleClient(llm_settings)
                        st.write(f"正在调用 {client.model}")
                        result = LLMPolicyAdjudicator(
                            store,
                            client,
                            additional_instructions=prompt_instructions,
                        ).adjudicate(
                            case_id=case_id,
                            content_type=content_type,
                            input_text=input_text,
                            images=images,
                        )
                        engine_name = client.model
                    else:
                        st.write("正在执行结构化规则、品牌匹配与政策检索")
                        result = ProductReviewAssistant(library, store).review(product_item)
                        engine_name = "structured-rules-v2"
                    st.session_state["workbench_result_kind"] = (
                        "product" if isinstance(result, ProductReviewResult) else "decision"
                    )
                    st.session_state["workbench_result"] = result
                    st.session_state["workbench_engine"] = engine_name
                    st.session_state["workbench_image_used"] = bool(images)
                    st.session_state["workbench_skill"] = (
                        f"{resolved.skill.skill_id}@{resolved.skill.version}"
                        if resolved
                        else "离线基线"
                    )
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                st.session_state["workbench_mode_used"] = mode
                st.session_state["workbench_elapsed_ms"] = elapsed_ms
                status.update(label="审核链路执行完成", state="complete", expanded=False)
        except ValueError as exc:
            with result_col:
                st.error(str(exc))

    with result_col, st.container(border=True):
        st.subheader("决策结果")
        result = st.session_state.get("workbench_result")
        if result is None:
            st.markdown(
                """
                <div class="lab-empty">
                    <strong>选择一个演示案例并开始审核</strong>
                    结果、政策证据、工具调用和安全门禁会集中显示在这里。
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        result_kind = st.session_state.get("workbench_result_kind")
        if result_kind == "agent":
            render_agent_result(result)
        elif result_kind == "product":
            render_product_result(result)
        else:
            render_decision(
                result,
                st.session_state.get("workbench_engine", "unknown"),
                st.session_state.get("workbench_image_used", False),
            )
        st.caption(
            f"运行配置：`{st.session_state.get('workbench_skill', 'unknown')}` · "
            f"耗时 `{st.session_state.get('workbench_elapsed_ms', 0)} ms`"
        )
        with st.expander("执行链路", expanded=True):
            render_execution_steps(
                st.session_state.get("workbench_mode_used", mode), result
            )
        if result_kind == "agent":
            with st.expander("工具调用详情"):
                for item in result.tool_executions:
                    st.markdown(f"**步骤 {item.step} · `{item.tool_name}`**")
                    st.json(item.output)


def render_case_workspace(
    store: PolicyVectorStore,
    llm_settings: ResilientLLMSettings,
) -> None:
    st.subheader("案例审核")
    prompt_registry = get_prompt_skill_registry()
    general_skills = prompt_registry.skills(PromptWorkflow.GENERAL_CASE)
    with st.form("case-adjudication"):
        engine = st.segmented_control(
            "推理引擎",
            options=["多模态模型", "离线规则"],
            default="多模态模型",
        )
        content_type_labels = {
            ContentType.PRODUCT: "商品",
            ContentType.VIDEO: "视频",
            ContentType.SHOP_NAME: "店铺名称",
            ContentType.SHOP_AVATAR: "店铺头像",
            ContentType.QUERY: "搜索查询",
        }
        content_type = st.selectbox(
            "内容类型",
            options=list(ContentType),
            index=list(ContentType).index(ContentType.PRODUCT),
            format_func=lambda value: content_type_labels[value],
        )
        input_text = st.text_area(
            "待审核内容",
            value="Gucci 1:1 mirror copy handbag, includes branded dust bag",
            height=120,
            max_chars=2000,
        )
        selected_skill = st.selectbox(
            "审核 Skill",
            options=general_skills,
            format_func=lambda item: f"{item.name} · {item.version}",
            disabled=engine == "离线规则",
        )
        resolved_prompt = prompt_registry.resolve_skill(
            selected_skill.skill_id, selected_skill.version
        ).prompt
        customize_prompt = st.checkbox(
            "临时调整本次 Prompt",
            disabled=engine == "离线规则",
        )
        prompt_instructions = st.text_area(
            "Prompt 指令",
            value=resolved_prompt.instructions,
            height=170,
            disabled=engine == "离线规则" or not customize_prompt,
        )
        if engine == "离线规则":
            st.caption("离线规则不调用模型，因此不会使用 Prompt 或 Skill。")
        image_col, url_col = st.columns(2)
        with image_col:
            uploaded_image = st.file_uploader(
                "上传图片",
                type=["png", "jpg", "jpeg", "webp"],
                key="case_image",
            )
        with url_col:
            image_url = st.text_input(
                "图片 URL",
                placeholder="https://example.com/product.jpg",
            )
        if uploaded_image:
            st.image(uploaded_image, width=280)
        submitted = st.form_submit_button(
            ":material/gavel: 开始审核",
            type="primary",
        )

    if submitted:
        images = []
        try:
            if uploaded_image:
                images.append(
                    ImageInput(
                        data=uploaded_image.getvalue(),
                        media_type=uploaded_image.type or "image/jpeg",
                    )
                )
            elif image_url.strip():
                images.append(ImageInput(url=image_url.strip()))

            if engine == "离线规则":
                decision = PolicyAdjudicator(store).adjudicate(
                    case_id="C000",
                    content_type=content_type,
                    input_text=input_text,
                )
                engine_name = "offline-rules-v1"
            else:
                if not llm_settings.configured:
                    raise ValueError("请先配置 API Key 和视觉模型。")
                client = ResilientOpenAICompatibleClient(llm_settings)
                with st.spinner(f"正在调用 {client.model}..."):
                    decision = LLMPolicyAdjudicator(
                        store,
                        client,
                        additional_instructions=prompt_instructions,
                    ).adjudicate(
                        case_id="C000",
                        content_type=content_type,
                        input_text=input_text,
                        images=images,
                    )
                engine_name = client.model
                st.session_state["last_prompt_version"] = (
                    f"{resolved_prompt.prompt_id}@{resolved_prompt.version}"
                )
                st.session_state["last_skill_version"] = (
                    f"{selected_skill.skill_id}@{selected_skill.version}"
                )
            st.session_state["last_decision"] = decision
            st.session_state["last_engine"] = engine_name
            st.session_state["last_image_used"] = bool(images)
        except ValueError as exc:
            st.error(str(exc))

    decision = st.session_state.get("last_decision")
    if decision:
        render_decision(
            decision,
            st.session_state.get("last_engine", "unknown"),
            st.session_state.get("last_image_used", False),
        )
        if st.session_state.get("last_skill_version") and engine != "离线规则":
            st.caption(
                "运行配置："
                f"`{st.session_state['last_skill_version']}` · "
                f"`{st.session_state['last_prompt_version']}`"
            )


def render_product_review(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
    trace_store: TraceStore,
) -> None:
    render_page_header(
        "决策工作台",
        "商品知识产权审核",
        "通过候选召回、受控品牌匹配、上下文校验和政策证据形成可审计建议。",
        [("确定性工作流", ""), ("品牌命中不等于违规", "")],
    )
    with st.form("product-review"):
        id_col, product_col = st.columns(2)
        with id_col:
            case_id = st.text_input("案例 ID", value="PR-001")
        with product_col:
            product_id = st.text_input("商品 ID", value="SKU-001")

        title = st.text_input("商品标题", value="Gucci 手提包")
        description = st.text_area(
            "商品描述",
            value="1:1 mirror copy，附带品牌防尘袋",
            height=110,
        )
        brand_col, category_col = st.columns(2)
        with brand_col:
            brand_field = st.text_input("卖家填写的品牌（可选）")
        with category_col:
            category = st.text_input("商品类目（可选）", value="箱包")

        image_col, url_col = st.columns(2)
        with image_col:
            product_image = st.file_uploader(
                "商品图片",
                type=["png", "jpg", "jpeg", "webp"],
                key="product_review_image",
            )
        with url_col:
            product_image_url = st.text_input(
                "商品图片 URL",
                placeholder="https://example.com/product.jpg",
            )
        if product_image:
            st.image(product_image, width=280)

        ocr_text = st.text_area("图片 OCR 文本（可选）", height=80)
        visual_marks = st.text_input(
            "视觉标记（逗号分隔）",
            placeholder="Gucci Logo, 品牌包装",
        )
        with st.expander("结构化业务证据", expanded=True):
            signal_col_one, signal_col_two = st.columns(2)
            brand_authorization_status = signal_col_one.selectbox(
                "品牌授权状态",
                options=list(BrandAuthorizationStatus),
                format_func=lambda value: {
                    BrandAuthorizationStatus.AUTHORIZED: "已授权",
                    BrandAuthorizationStatus.UNAUTHORIZED: "未授权",
                    BrandAuthorizationStatus.UNKNOWN: "未知",
                }[value],
            )
            observed_product_brand = signal_col_two.text_input(
                "图片中观察到的商品品牌"
            )
            logo_match_type = signal_col_one.selectbox(
                "Logo 匹配", options=list(LogoMatchType)
            )
            design_similarity = signal_col_two.selectbox(
                "外观相似度信号", options=list(DesignSimilaritySignal)
            )
            image_quality = signal_col_one.selectbox(
                "图片质量", options=list(ImageQuality)
            )
            visual_confidence = signal_col_two.slider(
                "视觉证据置信度", 0.0, 1.0, 1.0, 0.05
            )
            independent_brand_registered = st.checkbox(
                "已提供独立品牌注册证据"
            )
            price_col_one, price_col_two = st.columns(2)
            listing_price = price_col_one.number_input(
                "商品价格（0 表示未提供）", min_value=0.0, value=0.0
            )
            reference_price = price_col_two.number_input(
                "参考价格（0 表示未提供）", min_value=0.0, value=0.0
            )
        submitted = st.form_submit_button(
            ":material/fact_check: 开始商品审核",
            type="primary",
        )

    if submitted:
        try:
            if product_image and product_image_url.strip():
                raise ValueError("请只选择一种图片来源。")
            image_path = None
            if product_image:
                image_path = f"uploads/{Path(product_image.name).name}"
            item = ProductReviewInput(
                case_id=case_id,
                product_id=product_id or None,
                title=title,
                description=description,
                optional_brand_field=brand_field or None,
                optional_category=category or None,
                image_url=product_image_url.strip() or None,
                image_path=image_path,
                ocr_text=ocr_text,
                visual_marks=[
                    mark.strip()
                    for mark in visual_marks.replace("，", ",").split(",")
                    if mark.strip()
                ],
                brand_authorization_status=brand_authorization_status,
                observed_product_brand=observed_product_brand or None,
                logo_match_type=logo_match_type,
                design_similarity=design_similarity,
                independent_brand_registered=independent_brand_registered,
                listing_price=listing_price or None,
                reference_price=reference_price or None,
                image_quality=image_quality,
                visual_confidence=visual_confidence,
            )
            started = time.perf_counter()
            result = ProductReviewAssistant(library, store).review(item)
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            trace_store.append(product_review_trace(item, result, latency_ms))
            st.session_state["last_product_review"] = result
            st.session_state["last_product_input"] = item
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("last_product_review")
    if not result:
        return

    status_labels = {
        ProductReviewDecision.APPROVE: "通过",
        ProductReviewDecision.REJECT: "拒绝",
        ProductReviewDecision.MANUAL_REVIEW: "人工复核",
    }
    status = status_labels[result.suggested_decision]
    if result.suggested_decision == ProductReviewDecision.REJECT:
        st.error(f"建议结果：{status}")
    elif result.suggested_decision == ProductReviewDecision.APPROVE:
        st.success(f"建议结果：{status}")
    else:
        st.warning(f"建议结果：{status}")

    summary_cols = st.columns(3)
    summary_cols[0].metric("置信度", result.confidence.value)
    summary_cols[1].metric("受控品牌", len(result.controlled_brand_matches))
    summary_cols[2].metric("策略证据", len(result.policy_references))
    st.markdown(f"**判断理由**  \n{result.reason}")

    if result.candidate_brands:
        st.markdown("**品牌候选与上下文校验**")
        st.dataframe(
            [
                {
                    "候选品牌": candidate.candidate_brand,
                    "受控品牌": candidate.matched_brand or "未命中",
                    "来源": candidate.source.value,
                    "上下文置信度": candidate.context_confidence.value,
                    "是否歧义": "是" if candidate.ambiguity else "否",
                    "证据": candidate.evidence,
                    "说明": candidate.notes,
                }
                for candidate in result.candidate_brands
            ],
            width="stretch",
            hide_index=True,
        )

    if result.policy_references:
        st.markdown("**策略引用**")
        for reference in result.policy_references:
            with st.expander(
                f"{reference.policy_id} · {reference.title} · "
                f"相关度 {reference.retrieval_score:.3f}",
                expanded=True,
            ):
                st.markdown(reference.quote)

    if result.reviewer_checkpoints:
        st.markdown("**人工复核要点**")
        for checkpoint in result.reviewer_checkpoints:
            st.markdown(f"- {checkpoint}")

    with st.expander("结构化输出"):
        st.code(result.model_dump_json(indent=2), language="json")

    if st.button("加入人工审核队列", key=f"queue-product-{result.case_id}"):
        try:
            item = st.session_state["last_product_input"]
            review_store.enqueue(
                case_id=result.case_id,
                workflow=ReviewWorkflow.PRODUCT_IPR,
                case_input=item.model_dump(mode="json"),
                agent_decision=result.suggested_decision,
                agent_confidence=result.confidence,
                agent_reason=result.reason,
                evidence=[value.model_dump(mode="json") for value in result.evidence],
                policy_evidence=[value.model_dump(mode="json") for value in result.policy_references],
            )
            st.success("已加入人工审核队列。")
        except (KeyError, ValueError) as exc:
            st.error(str(exc))


def render_shop_result(result) -> None:
    status_labels = {
        ProductReviewDecision.APPROVE: "通过",
        ProductReviewDecision.REJECT: "拒绝",
        ProductReviewDecision.MANUAL_REVIEW: "人工复核",
    }
    status = status_labels[result.suggested_decision]
    if result.suggested_decision == ProductReviewDecision.REJECT:
        st.error(f"建议结果：{status}")
    elif result.suggested_decision == ProductReviewDecision.APPROVE:
        st.success(f"建议结果：{status}")
    else:
        st.warning(f"建议结果：{status}")
    columns = st.columns(3)
    columns[0].metric("置信度", result.confidence.value)
    columns[1].metric("店名信号", result.shop_name_signal.signal_type.value)
    columns[2].metric("头像信号", result.avatar_signal.signal_type.value)
    st.markdown(f"**判断理由**  \n{result.reason}")
    if result.possible_exemptions:
        st.markdown(f"**可能豁免**  \n{', '.join(result.possible_exemptions)}")
    if result.reviewer_checkpoints:
        st.markdown("**人工复核要点**")
        for checkpoint in result.reviewer_checkpoints:
            st.markdown(f"- {checkpoint}")
    if result.policy_references:
        st.markdown("**策略引用**")
        for reference in result.policy_references:
            with st.expander(f"{reference.policy_id} · {reference.title}"):
                st.markdown(reference.quote)
    with st.expander("结构化输出"):
        st.code(result.model_dump_json(indent=2), language="json")


def render_shop_identity(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
    trace_store: TraceStore,
) -> None:
    render_page_header(
        "决策工作台",
        "店铺身份审核",
        "优先检查授权状态，再评估店名与头像中的品牌冒充风险。",
        [("授权 Gate", ""), ("证据不足转人工", "")],
    )
    shop_demo_cases = {
        f"V5 · {item.name}": item
        for item in load_synthetic_demo_cases(DEFAULT_SYNTHETIC_DEMO_CASES_PATH)
        if item.workflow == DemoWorkflow.SHOP_IDENTITY
    }
    shop_demo_name = st.selectbox("演示案例", options=list(shop_demo_cases))
    shop_demo = shop_demo_cases[shop_demo_name]
    shop_demo_image = PROJECT_ROOT / shop_demo.image_path
    st.image(
        str(shop_demo_image),
        caption="合成店铺头像，不包含真实品牌或公司素材",
        width=240,
    )
    with st.form("shop-identity"):
        id_col, brand_col = st.columns(2)
        with id_col:
            case_id = st.text_input(
                "案例 ID", value=shop_demo.case_id, key=f"shop_case_id_{shop_demo.case_id}"
            )
        with brand_col:
            controlled_brand = st.text_input(
                "受控品牌", value=shop_demo.controlled_brand or ""
            )
        shop_name = st.text_input("店铺名称", value=shop_demo.shop_name or "")
        avatar_url = st.text_input("头像 URL（可选）")
        authorization = st.selectbox(
            "授权状态",
            options=list(AuthorizationStatus),
            index=list(AuthorizationStatus).index(
                AuthorizationStatus(shop_demo.brand_authorization_status.value)
            ),
            format_func=lambda value: {
                AuthorizationStatus.AUTHORIZED: "已授权",
                AuthorizationStatus.UNAUTHORIZED: "未授权",
                AuthorizationStatus.UNKNOWN: "未知",
            }[value],
        )
        avatar_marks = st.text_input(
            "头像视觉标记（逗号分隔）",
            value=", ".join(shop_demo.visual_marks),
            placeholder="品牌 logo, 官方字样",
        )
        submitted = st.form_submit_button(
            ":material/storefront: 开始店铺审核",
            type="primary",
        )
    if submitted:
        try:
            item = ShopIdentityInput(
                case_id=case_id,
                shop_name=shop_name,
                avatar_url=avatar_url.strip() or None,
                controlled_brand=controlled_brand,
                authorization_status=authorization,
                avatar_visual_marks=[
                    value.strip()
                    for value in avatar_marks.replace("，", ",").split(",")
                    if value.strip()
                ],
            )
            started = time.perf_counter()
            result = ShopIdentityReviewer(library, store).review(item)
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            trace_store.append(shop_identity_trace(item, result, latency_ms))
            st.session_state["last_shop_input"] = item
            st.session_state["last_shop_review"] = result
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("last_shop_review")
    if not result:
        return
    render_shop_result(result)
    if st.button("加入人工审核队列", key=f"queue-shop-{result.case_id}"):
        try:
            item = st.session_state["last_shop_input"]
            review_store.enqueue(
                case_id=result.case_id,
                workflow=ReviewWorkflow.SHOP_IDENTITY,
                case_input=item.model_dump(mode="json"),
                agent_decision=result.suggested_decision,
                agent_confidence=result.confidence,
                agent_reason=result.reason,
                evidence=[
                    result.shop_name_signal.model_dump(mode="json"),
                    result.avatar_signal.model_dump(mode="json"),
                ],
                policy_evidence=[value.model_dump(mode="json") for value in result.policy_references],
            )
            st.success("已加入人工审核队列。")
        except (KeyError, ValueError) as exc:
            st.error(str(exc))


def render_batch_workspace(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
    trace_store: TraceStore,
) -> None:
    render_page_header(
        "决策工作台",
        "批量审核",
        "导入 CSV 或 Excel，完成字段校验、批量决策、筛选与结果导出。",
        [("CSV / Excel", ""), ("结构化校验", "")],
    )
    workflow = st.segmented_control(
        "批量类型",
        options=["商品知识产权", "店铺身份"],
        default="商品知识产权",
    )
    uploaded = st.file_uploader(
        "上传 CSV 或 Excel",
        type=["csv", "xlsx"],
        key="batch_upload",
    )
    if st.button(":material/play_arrow: 运行批量校验", type="primary"):
        if uploaded is None:
            st.error("请先上传批量文件。")
        else:
            try:
                inputs = (
                    parse_product_batch(uploaded.getvalue(), uploaded.name)
                    if workflow == "商品知识产权"
                    else parse_shop_batch(uploaded.getvalue(), uploaded.name)
                )
                results = []
                for item in inputs:
                    started = time.perf_counter()
                    if workflow == "商品知识产权":
                        result = ProductReviewAssistant(library, store).review(item)
                        trace = product_review_trace(
                            item,
                            result,
                            max(0, round((time.perf_counter() - started) * 1000)),
                        )
                    else:
                        result = ShopIdentityReviewer(library, store).review(item)
                        trace = shop_identity_trace(
                            item,
                            result,
                            max(0, round((time.perf_counter() - started) * 1000)),
                        )
                    results.append(result)
                    trace_store.append(trace)
                st.session_state["batch_inputs"] = inputs
                st.session_state["batch_results"] = results
                st.session_state["batch_workflow"] = workflow
            except BatchValidationError as exc:
                for error in exc.errors:
                    st.error(error)

    results = st.session_state.get("batch_results", [])
    if not results:
        return
    decision_filter = st.multiselect(
        "结论筛选",
        options=["approve", "reject", "manual_review"],
        default=["approve", "reject", "manual_review"],
    )
    confidence_filter = st.multiselect(
        "置信度筛选",
        options=["high", "medium", "low"],
        default=["high", "medium", "low"],
    )
    visible = [
        result
        for result in results
        if result.suggested_decision.value in decision_filter
        and result.confidence.value in confidence_filter
    ]
    st.dataframe(
        [
            {
                "案例 ID": result.case_id,
                "建议结果": result.suggested_decision.value,
                "置信度": result.confidence.value,
                "理由": result.reason,
                "复核要点": " | ".join(result.reviewer_checkpoints),
            }
            for result in visible
        ],
        width="stretch",
        hide_index=True,
    )
    csv_col, xlsx_col = st.columns(2)
    csv_col.download_button(
        ":material/download: 导出 CSV",
        data=export_csv(visible),
        file_name="review_results.csv",
        mime="text/csv",
        disabled=not visible,
    )
    xlsx_col.download_button(
        ":material/download: 导出 Excel",
        data=export_xlsx(visible),
        file_name="review_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not visible,
    )
    if st.button("将人工复核项加入队列"):
        inputs_by_id = {item.case_id: item for item in st.session_state["batch_inputs"]}
        added = 0
        for result in results:
            if result.suggested_decision != ProductReviewDecision.MANUAL_REVIEW:
                continue
            item = inputs_by_id[result.case_id]
            workflow_value = (
                ReviewWorkflow.PRODUCT_IPR
                if st.session_state["batch_workflow"] == "商品知识产权"
                else ReviewWorkflow.SHOP_IDENTITY
            )
            try:
                review_store.enqueue(
                    case_id=result.case_id,
                    workflow=workflow_value,
                    case_input=item.model_dump(mode="json"),
                    agent_decision=result.suggested_decision,
                    agent_confidence=result.confidence,
                    agent_reason=result.reason,
                    evidence=[value.model_dump(mode="json") for value in getattr(result, "evidence", [])],
                    policy_evidence=[value.model_dump(mode="json") for value in result.policy_references],
                )
                added += 1
            except ValueError:
                continue
        st.success(f"已新增 {added} 条人工审核任务。")


def render_reviewer_workspace(
    review_store: ReviewerWorkspaceStore,
    staging_store: StagingDatasetStore,
) -> None:
    render_page_header(
        "决策工作台",
        "人工复核",
        "分别保留 Agent、Reviewer 与 Final Decision，并将确认案例送入 staging dataset。",
        [("人工反馈闭环", ""), ("Golden Set 隔离", "")],
    )
    records = review_store.list_records()
    if not records:
        st.info("当前没有人工审核任务。")
        return
    st.dataframe(
        [
            {
                "Review ID": record.review_id,
                "Case ID": record.case_id,
                "工作流": record.workflow.value,
                "Agent Decision": record.agent_decision.value,
                "Reviewer Decision": record.reviewer_label.value if record.reviewer_label else "pending",
                "Final Decision": record.final_decision.value if record.final_decision else "pending",
            }
            for record in records
        ],
        width="stretch",
        hide_index=True,
    )
    pending = [record for record in records if record.reviewer_label is None]
    if not pending:
        st.success("所有任务均已完成人工审核。")
    else:
        selected_id = st.selectbox(
            "待复核任务",
            options=[record.review_id for record in pending],
            format_func=lambda value: next(
                f"{item.case_id} · {item.workflow.value} · {item.agent_decision.value}"
                for item in pending
                if item.review_id == value
            ),
        )
        selected = next(item for item in pending if item.review_id == selected_id)
        columns = st.columns(2)
        with columns[0]:
            st.markdown("**Case Input**")
            st.json(selected.case_input)
        with columns[1]:
            st.markdown("**Agent Decision**")
            st.json(
                {
                    "decision": selected.agent_decision.value,
                    "confidence": selected.agent_confidence.value,
                    "reason": selected.agent_reason,
                }
            )
        with st.form("reviewer-feedback"):
            reviewer_label = st.segmented_control(
                "Reviewer Label",
                options=list(ReviewerLabel),
                default=ReviewerLabel.UNCERTAIN,
                format_func=lambda value: {
                    ReviewerLabel.APPROVE: "通过",
                    ReviewerLabel.REJECT: "拒绝",
                    ReviewerLabel.UNCERTAIN: "不确定",
                }[value],
            )
            reviewer_note = st.text_area("Reviewer Note")
            expected_policy = st.text_input("预期策略（可选）")
            case_category = st.text_input("案例分类", value=selected.workflow.value)
            failure_type = st.selectbox(
                "Failure Type",
                options=[
                    "unknown", "input_missing", "candidate_recall_miss", "retrieval_miss",
                    "evidence_miss", "exemption_miss", "policy_misread", "schema_error",
                    "routing_error", "low_confidence",
                ],
            )
            add_to_golden = st.checkbox("加入 Golden Set staging")
            save = st.form_submit_button(":material/save: 保存人工结论", type="primary")
        if save:
            try:
                completed = review_store.complete(
                    selected.review_id,
                    ReviewerFeedback(
                        reviewer_label=reviewer_label,
                        reviewer_note=reviewer_note,
                        add_to_golden_set=add_to_golden,
                        expected_policy=expected_policy or None,
                        case_category=case_category,
                        failure_type=failure_type,
                    ),
                )
                if completed.add_to_golden_set:
                    staging_store.add_review(completed)
                st.success("人工结论已保存，Agent Decision 已保留。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    candidates = staging_store.list_records()
    st.divider()
    st.markdown(f"**Staging Dataset** · {len(candidates)} 条")
    if candidates:
        st.dataframe(
            [
                {
                    "Candidate ID": item.candidate_id,
                    "Case ID": item.case_id,
                    "预期结论": item.expected_decision,
                    "Failure Type": item.failure_type,
                    "状态": item.status.value,
                }
                for item in candidates
            ],
            width="stretch",
            hide_index=True,
        )
        if st.button("显式 Promote 为 Golden Set v5"):
            try:
                manifest = staging_store.promote(
                    DEFAULT_STAGING_DATASET_PATH.parent.parent / "v5"
                )
                st.success(
                    f"已生成 {manifest.data_file}，共 {manifest.record_count} 条，"
                    f"SHA-256: {manifest.sha256[:12]}..."
                )
            except ValueError as exc:
                st.error(str(exc))


def render_trace_workspace(trace_store: TraceStore) -> None:
    render_page_header(
        "AI 实验室",
        "Trace 与失败分析",
        "检查每个工作流节点的输入摘要、输出、耗时、Token 与失败原因。",
        [("敏感字段过滤", "lab-chip--ok"), ("Failure Taxonomy", "")],
    )
    traces = trace_store.list_traces()
    if not traces:
        st.info("运行商品、店铺或批量审核后，Trace 将显示在这里。")
        return
    case_ids = sorted({trace.case_id for trace in traces})
    selected_case = st.selectbox("Case ID", case_ids)
    case_traces = [trace for trace in traces if trace.case_id == selected_case]
    trace = case_traces[-1]
    st.caption(f"Trace ID: `{trace.trace_id}` · Workflow: `{trace.workflow}`")
    st.dataframe(
        [
            {
                "节点": node.node_name,
                "状态": node.status.value,
                "耗时 ms": node.latency_ms,
                "置信度": node.confidence,
                "Token": node.token_usage.total_tokens if node.token_usage else 0,
                "错误": node.error or "",
            }
            for node in trace.nodes
        ],
        width="stretch",
        hide_index=True,
    )
    for node in trace.nodes:
        with st.expander(f"{node.node_name} · {node.status.value}"):
            st.markdown("**Input Summary**")
            st.json(node.input_summary)
            st.markdown("**Output**")
            st.json(node.output)
    with st.expander("Final Output"):
        st.json(trace.final_output)


def render_controlled_agent_workspace(
    store: PolicyVectorStore,
    library: ControlledBrandLibrary,
    review_store: ReviewerWorkspaceStore,
    llm_settings: ResilientLLMSettings,
) -> None:
    st.subheader("受控 Tool Calling Agent")
    st.caption("模型负责选择只读工具，代码负责权限、步数、参数与最终决策门禁。")
    prompt_registry = get_prompt_skill_registry()
    agent_skills = prompt_registry.skills(PromptWorkflow.CONTROLLED_AGENT)
    with st.form("controlled-agent-form"):
        selected_skill = st.selectbox(
            "Agent Skill",
            options=agent_skills,
            format_func=lambda item: f"{item.name} · {item.version}",
        )
        resolved_skill = prompt_registry.resolve_skill(
            selected_skill.skill_id, selected_skill.version
        )
        first, second = st.columns(2)
        case_id = first.text_input("Case ID", value="AGENT-DEMO-001")
        title = second.text_input("商品标题", value="Gucci 手提包")
        description = st.text_area("商品描述", value="1:1 mirror copy，非专柜正品")
        third, fourth = st.columns(2)
        brand = third.text_input("卖家品牌字段", value="Gucci")
        ocr_text = fourth.text_input("OCR 文本", value="Gucci")
        task = st.text_area(
            "Agent 任务",
            value="调用必要工具完成商品知识产权审核；证据不足时转人工。",
        )
        customize_prompt = st.checkbox("临时调整本次 Prompt")
        agent_instructions = st.text_area(
            "Prompt 指令",
            value=resolved_skill.prompt.instructions,
            height=180,
            disabled=not customize_prompt,
        )
        run_agent = st.form_submit_button(
            ":material/play_arrow: 运行 Agent",
            type="primary",
        )

    if run_agent:
        if not llm_settings.configured:
            st.error("请先在侧边栏配置模型 API Key 与模型名称。")
        else:
            case_data = {
                "case_id": case_id,
                "title": title,
                "description": description,
                "optional_brand_field": brand or None,
                "ocr_text": ocr_text,
                "visual_marks": [brand] if brand else [],
            }
            try:
                registry = build_default_tool_registry(store, library, review_store)
                planner = ResilientOpenAICompatibleClient(llm_settings)
                request = AgentRequest(case_id=case_id, task=task, case_data=case_data)
                result = ControlledPolicyAgent(
                    planner,
                    registry,
                    max_steps=resolved_skill.skill.max_steps,
                    allowed_tools=resolved_skill.skill.allowed_tools,
                    additional_instructions=agent_instructions,
                    policy_scope=resolved_skill.skill.policy_scope,
                ).run(request)
                get_trace_store().append(controlled_agent_trace(request, result))
                st.session_state["controlled_agent_result"] = result
                st.session_state["controlled_agent_case"] = case_data
                st.session_state["controlled_agent_skill"] = (
                    f"{resolved_skill.skill.skill_id}@{resolved_skill.skill.version}"
                )
                st.session_state["controlled_agent_prompt"] = (
                    f"{resolved_skill.prompt.prompt_id}@{resolved_skill.prompt.version}"
                )
            except ValueError as exc:
                st.error(str(exc))

    result = st.session_state.get("controlled_agent_result")
    if result is None:
        st.info("运行后会在这里展示工具调用轨迹和最终门禁结果。")
        return
    metrics = st.columns(4)
    metrics[0].metric("最终建议", result.decision.value)
    metrics[1].metric("置信度", result.confidence.value)
    metrics[2].metric("工具调用", len(result.tool_executions))
    metrics[3].metric("Token", result.total_tokens)
    st.markdown(f"**停止原因：** `{result.stop_reason.value}`")
    st.caption(
        "运行配置："
        f"`{st.session_state.get('controlled_agent_skill', 'unknown')}` · "
        f"`{st.session_state.get('controlled_agent_prompt', 'unknown')}`"
    )
    st.write(result.reason)
    st.dataframe(
        [
            {
                "步骤": item.step,
                "工具": item.tool_name,
                "权限": item.permission.value,
                "状态": item.status.value,
                "耗时 ms": item.latency_ms,
                "错误": item.error or "",
            }
            for item in result.tool_executions
        ],
        width="stretch",
        hide_index=True,
    )
    for item in result.tool_executions:
        with st.expander(f"步骤 {item.step} · {item.tool_name}"):
            st.markdown("**参数**")
            st.json(item.arguments)
            st.markdown("**结构化输出**")
            st.json(item.output)
    if result.decision == ProductReviewDecision.MANUAL_REVIEW:
        if st.button(":material/person_add: 确认加入人工复核队列"):
            try:
                review_store.enqueue(
                    case_id=result.case_id,
                    workflow=ReviewWorkflow.GENERAL_CASE,
                    case_input=st.session_state.get("controlled_agent_case", {}),
                    agent_decision=result.decision,
                    agent_confidence=result.confidence,
                    agent_reason=result.reason,
                    evidence=[
                        item.model_dump(mode="json")
                        for item in result.tool_executions
                    ],
                )
                st.success("已在用户确认后创建人工复核任务。")
            except ValueError as exc:
                st.error(str(exc))


def render_prompt_skill_studio() -> None:
    render_page_header(
        "AI 实验室",
        "Prompt 与 Skill",
        "Prompt 定义模型判断指令；Skill 将 Prompt、工具权限、政策范围、步数和输出契约封装为可复用版本。",
        [("版本化配置", ""), ("运行边界不可覆盖", "lab-chip--ok")],
    )
    registry = get_prompt_skill_registry()
    st.info("这里保存的是版本化资产，不会自动改变审核端。请在“Workflow 管理”中完成评测后的显式发布或回退。")
    overview_tab, revise_tab, prompt_tab, skill_tab = st.tabs(
        ["查看版本", "基于版本修改", "新建 Prompt", "新建 Skill"]
    )

    with overview_tab:
        prompt_detail_tab, skill_detail_tab = st.tabs(["Prompt 详情", "Skill 详情"])
        prompts = registry.prompts()
        skills = registry.skills()
        with prompt_detail_tab:
            st.dataframe([{"Prompt ID": item.prompt_id, "版本": item.version, "名称": item.name, "工作流": item.workflow.value, "变更说明": item.change_note} for item in prompts], width="stretch", hide_index=True)
            selected_prompt = st.selectbox("选择 Prompt 版本", prompts, index=len(prompts) - 1, format_func=lambda item: f"{item.name} · {item.prompt_id}@{item.version}")
            st.caption(f"工作流：`{selected_prompt.workflow.value}` · 变更说明：{selected_prompt.change_note}")
            st.text_area("Prompt 完整内容", selected_prompt.instructions, height=420, disabled=True, key="prompt-version-content")
        with skill_detail_tab:
            st.dataframe([{"Skill ID": item.skill_id, "版本": item.version, "名称": item.name, "工作流": item.workflow.value, "Prompt": f"{item.prompt_id}@{item.prompt_version}", "工具": ", ".join(item.allowed_tools) or "无", "政策范围": ", ".join(item.policy_scope) or "全部", "最大步骤": item.max_steps} for item in skills], width="stretch", hide_index=True)
            selected_skill = st.selectbox("选择 Skill 版本", skills, index=len(skills) - 1, format_func=lambda item: f"{item.name} · {item.skill_id}@{item.version}")
            resolved = registry.resolve_skill(selected_skill.skill_id, selected_skill.version)
            left, right = st.columns(2)
            left.write({"Skill ID": selected_skill.skill_id, "版本": selected_skill.version, "名称": selected_skill.name, "工作流": selected_skill.workflow.value, "最大步骤": selected_skill.max_steps, "输出契约": selected_skill.output_contract})
            right.write({"绑定 Prompt": f"{selected_skill.prompt_id}@{selected_skill.prompt_version}", "允许工具": selected_skill.allowed_tools or ["无"], "政策范围": selected_skill.policy_scope or ["全部"]})
            st.markdown("**Skill 说明**")
            st.write(selected_skill.description)
            st.text_area("绑定的 Prompt 完整内容", resolved.prompt.instructions, height=360, disabled=True, key="skill-linked-prompt-content")

    with revise_tab:
        st.caption("修改会生成同一 ID 的新版本，不覆盖原版本；保存后仍需评测和显式发布。")
        revise_prompt_tab, revise_skill_tab = st.tabs(["修改 Prompt", "修改 Skill"])
        with revise_prompt_tab:
            base_prompt = st.selectbox("选择作为基底的 Prompt", registry.prompts(), index=len(registry.prompts()) - 1, format_func=lambda item: f"{item.name} · {item.prompt_id}@{item.version}", key="revise-prompt-base")
            next_prompt_version = registry.next_patch_version(base_prompt.version)
            with st.form("revise-prompt-version"):
                st.text_input("新版本", value=next_prompt_version, disabled=True)
                revised_prompt_name = st.text_input("名称", value=base_prompt.name)
                revised_prompt_text = st.text_area("Prompt 内容", value=base_prompt.instructions, height=420)
                revised_prompt_note = st.text_input("变更说明", value="基于现有版本调整")
                save_revised_prompt = st.form_submit_button("保存为新 Prompt 版本", type="primary")
            if save_revised_prompt:
                try:
                    revised = registry.revise_prompt(base_prompt, name=revised_prompt_name.strip(), instructions=revised_prompt_text.strip(), change_note=revised_prompt_note.strip())
                    st.success(f"已生成 `{revised.prompt_id}@{revised.version}`，原版本保持不变。")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with revise_skill_tab:
            all_skills = registry.skills()
            base_skill = st.selectbox("选择作为基底的 Skill", all_skills, index=len(all_skills) - 1, format_func=lambda item: f"{item.name} · {item.skill_id}@{item.version}", key="revise-skill-base")
            compatible_prompts = registry.prompts(base_skill.workflow)
            current_prompt_index = next((index for index, item in enumerate(compatible_prompts) if item.prompt_id == base_skill.prompt_id and item.version == base_skill.prompt_version), len(compatible_prompts) - 1)
            with st.form("revise-skill-version"):
                st.text_input("新版本", value=registry.next_patch_version(base_skill.version), disabled=True, key="revised-skill-version")
                revised_skill_name = st.text_input("名称", value=base_skill.name, key="revised-skill-name")
                revised_skill_description = st.text_area("说明", value=base_skill.description, key="revised-skill-description")
                revised_linked_prompt = st.selectbox("绑定 Prompt 版本", compatible_prompts, index=current_prompt_index, format_func=lambda item: f"{item.name} · {item.prompt_id}@{item.version}")
                tool_options = ["classify_policy_signals", "lookup_brand", "review_product", "search_policy"]
                revised_tools = st.multiselect("允许工具", tool_options, default=[item for item in base_skill.allowed_tools if item in tool_options])
                revised_scope = st.text_input("政策范围（逗号分隔）", value=", ".join(base_skill.policy_scope))
                revised_steps = st.number_input("最大步骤", min_value=1, max_value=8, value=base_skill.max_steps)
                revised_contract = st.text_input("输出契约", value=base_skill.output_contract)
                save_revised_skill = st.form_submit_button("保存为新 Skill 版本", type="primary")
            if save_revised_skill:
                try:
                    revised = registry.revise_skill(base_skill, name=revised_skill_name.strip(), description=revised_skill_description.strip(), prompt_id=revised_linked_prompt.prompt_id, prompt_version=revised_linked_prompt.version, allowed_tools=revised_tools, policy_scope=[item.strip() for item in revised_scope.split(",") if item.strip()], max_steps=int(revised_steps), output_contract=revised_contract.strip())
                    st.success(f"已生成 `{revised.skill_id}@{revised.version}`，原版本保持不变。")
                    st.rerun()
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

    with prompt_tab:
        with st.form("create-prompt-version"):
            prompt_id = st.text_input("Prompt ID", value="PRM-CUSTOM-REVIEW")
            prompt_version = st.text_input("版本", value="v1.0.0")
            prompt_name = st.text_input("名称", value="自定义审核 Prompt")
            prompt_workflow = st.selectbox(
                "适用工作流",
                list(PromptWorkflow),
                format_func=lambda value: {
                    PromptWorkflow.GENERAL_CASE: "案例审核",
                    PromptWorkflow.CONTROLLED_AGENT: "受控 Agent",
                }[value],
            )
            prompt_text = st.text_area(
                "指令内容",
                value=(
                    "【角色】你是平台知识产权审核辅助 Agent，只提供审核建议。\n"
                    "【目标】根据政策证据识别 Counterfeit、Knockoff、MBA 与 TMI。\n"
                    "【输入】检查标题、描述、商品图片、品牌字段、授权状态和价格证据。\n"
                    "【步骤】先校验输入，再召回品牌与政策，逐项判断标签，最后执行 Guardrail。\n"
                    "【边界】品牌词、低价或通用外观不能单独构成违规；关键证据不足时转人工。\n"
                    "【Few-shot】标题为“兼容 Apple iPhone 的第三方手机壳”且无官方暗示时，适用兼容性豁免。\n"
                    "【输出】返回结构化标签、证据、政策引用、置信度和人工检查点。"
                ),
                height=300,
            )
            change_note = st.text_input("变更说明", value="首次创建")
            save_prompt = st.form_submit_button(
                ":material/save: 保存 Prompt 版本", type="primary"
            )
        if save_prompt:
            try:
                registry.save_prompt(
                    PromptVersion(
                        prompt_id=prompt_id.strip().upper(),
                        version=prompt_version.strip(),
                        name=prompt_name.strip(),
                        workflow=prompt_workflow,
                        instructions=prompt_text.strip(),
                        change_note=change_note.strip(),
                    )
                )
                st.success("Prompt 版本已保存。它尚未发布，不会自动影响审核端。")
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

    with skill_tab:
        available_prompts = registry.prompts()
        with st.form("create-skill-version"):
            skill_id = st.text_input("Skill ID", value="SKL-CUSTOM-REVIEW")
            skill_version = st.text_input("版本", value="v1.0.0", key="skill-version")
            skill_name = st.text_input("名称", value="自定义审核 Skill")
            skill_description = st.text_area(
                "说明", value="封装审核 Prompt、工具权限与运行边界。"
            )
            linked_prompt = st.selectbox(
                "绑定 Prompt",
                available_prompts,
                format_func=lambda item: f"{item.name} · {item.prompt_id}@{item.version}",
            )
            tool_options = [
                "classify_policy_signals",
                "lookup_brand",
                "review_product",
                "search_policy",
            ]
            skill_tools = st.multiselect(
                "允许工具",
                tool_options,
                default=(
                    tool_options if linked_prompt.workflow == PromptWorkflow.CONTROLLED_AGENT else []
                ),
                disabled=linked_prompt.workflow == PromptWorkflow.GENERAL_CASE,
            )
            policy_scope_text = st.text_input(
                "政策范围（逗号分隔，留空表示全部）",
                value="POL-KO-001, POL-EX-001",
            )
            max_steps = st.number_input("最大步骤", min_value=1, max_value=8, value=4)
            output_contract = st.text_input(
                "输出契约", value="strict_structured_decision_v1"
            )
            save_skill = st.form_submit_button(
                ":material/save: 保存 Skill 版本", type="primary"
            )
        if save_skill:
            try:
                registry.save_skill(
                    SkillDefinition(
                        skill_id=skill_id.strip().upper(),
                        version=skill_version.strip(),
                        name=skill_name.strip(),
                        description=skill_description.strip(),
                        workflow=linked_prompt.workflow,
                        prompt_id=linked_prompt.prompt_id,
                        prompt_version=linked_prompt.version,
                        allowed_tools=skill_tools,
                        policy_scope=[
                            item.strip()
                            for item in policy_scope_text.split(",")
                            if item.strip()
                        ],
                        max_steps=int(max_steps),
                        output_contract=output_contract.strip(),
                    )
                )
                st.success("Skill 版本已保存。请先验证效果，再到 Workflow 管理中显式发布。")
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_evaluation_lab() -> None:
    render_page_header(
        "质量闭环",
        "Workflow 效果评测",
        "用固定数据集检验一次 Workflow 配置：先运行，再读指标和失败样本，最后决定是否发布。",
        [("Golden Set v4", ""), ("可复现 Run", "lab-chip--ok")],
    )
    st.markdown("**评测动线**　① 固定数据集与配置　→　② 创建评测记录（Run）　→　③ 阅读质量与人工率　→　④ 查看失败原因　→　⑤ 对比候选版本并决定发布")
    with st.expander("这些指标分别是什么意思？"):
        st.markdown(
            """
- **自动准确率（Auto Accuracy）**：系统作出“通过/拒绝”自动判断的 Case 中，有多少判断正确。越高越好。
- **覆盖率（Coverage）**：全部 Case 中，有多少可以自动判断而不转人工。越高不一定越好，需要和错误率一起看。
- **分流准确率（Route Accuracy）**：该转人工的 Case 是否真的被送去人工、可自动处理的 Case 是否没有被误转。
- **误放率（False Approve）**：本应拒绝却被通过，是治理场景最需要关注的风险。
- **误杀率（False Reject）**：本应通过却被拒绝，会影响正常商家和审核体验。
- **失败原因分类（Failure Taxonomy）**：把错误按“证据没提取到、RAG 没召回、规则理解错误、输出格式错误”等环节归类，帮助定位该改数据、检索、Prompt 还是 Workflow。
- **质量门禁（Quality Gate）**：预先设定的最低上线标准；未通过时，候选配置不应发布。
"""
        )
    lab = EvaluationLabStore(DEFAULT_EVALUATION_RUNS_PATH)
    runs = lab.list_runs()
    run_tab, result_tab, compare_tab = st.tabs(["1. 创建评测", "2. 阅读结果", "3. 版本对比"])
    with run_tab:
        st.markdown("**当前固定条件**")
        st.dataframe([{"数据集": "Golden Set v4（冻结）", "执行引擎": "离线规则基线", "政策索引": get_store().collection_name, "说明": "同一配置可重复运行并复现结果"}], width="stretch", hide_index=True)
        run_id = st.text_input("评测记录 ID（Run ID）", value=f"EV-LAB-{time.strftime('%Y%m%d-%H%M%S')}", key="evaluation-lab-run-id")
        if st.button(":material/play_arrow: 开始评测", type="primary"):
            try:
                with st.spinner("正在用 Golden Set v4 逐条运行..."):
                    report = run_offline_rules_experiment(run_id=run_id, dataset_path=DEFAULT_GOLDEN_SET_V4_PATH, output_root=DEFAULT_EVALUATION_RUNS_PATH, store=get_store())
                st.success(f"评测完成：自动准确率 {report.metrics.auto_accuracy:.1%}，覆盖率 {report.metrics.coverage:.1%}。请进入“阅读结果”。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with result_tab:
        if not runs:
            st.info("还没有评测记录。请先在“创建评测”运行一次。")
        else:
            run_map = {run.run_id: run for run in runs}
            selected_id = st.selectbox("选择评测记录（Run）", list(run_map), index=len(run_map) - 1)
            report = run_map[selected_id].report
            overview = st.columns(5)
            overview[0].metric("自动准确率", f"{report.metrics.auto_accuracy:.1%}", help="自动作出通过/拒绝判断的 Case 中，判断正确的比例。")
            overview[1].metric("覆盖率", f"{report.metrics.coverage:.1%}", help="全部 Case 中没有转人工、由系统自动处理的比例。")
            overview[2].metric("分流准确率", f"{report.metrics.route_accuracy:.1%}", help="自动处理与转人工的分流是否符合标注预期。")
            overview[3].metric("人工复核率", f"{report.metrics.review_rate:.1%}", help="全部 Case 中被转给人工的比例。")
            overview[4].metric("质量门禁", "通过" if report.passed else "未通过", help="是否达到预设的最低发布条件。")
            st.caption(f"执行引擎 `{report.engine}` · Prompt `{report.prompt_version}` · 数据集 `{report.dataset_version}` · 检索版本 `{report.retrieval_version}`")
            if report.failed_gates:
                st.warning("未通过的发布条件：" + "；".join(report.failed_gates))
            if report.metrics.failure_counts:
                failure_names = {"evidence_miss": "证据提取遗漏", "retrieval_miss": "RAG 召回遗漏", "policy_misread": "政策理解错误", "routing_error": "自动/人工分流错误", "schema_error": "结构化输出错误", "exemption_miss": "豁免识别遗漏", "low_confidence": "置信度不足", "unknown": "暂未归因"}
                st.markdown("**失败原因分类（Failure Taxonomy）**")
                st.dataframe([{"失败原因": failure_names.get(name, name), "系统代码": name, "数量": count, "建议检查": {"evidence_miss": "输入字段、OCR 或视觉证据", "retrieval_miss": "RAG 查询与知识分块", "policy_misread": "Prompt、规则与 Few-shot", "routing_error": "转人工阈值和 Guardrail", "schema_error": "输出格式与模型兼容性", "exemption_miss": "豁免政策与示例"}.get(name, "进入 Bad Case 后人工确认")} for name, count in sorted(report.metrics.failure_counts.items()) if count], width="stretch", hide_index=True)
            st.info("下一步：先查看失败原因，再到 Bad Case 队列确认具体 Case；需要定位节点时进入 Trace 与重放。")
    with compare_tab:
        if len(runs) < 2:
            st.info("至少需要两个评测记录才能比较版本。Baseline 是当前基准，Candidate 是准备验证的新版本。")
        else:
            run_map = {run.run_id: run for run in runs}
            left, right = st.columns(2)
            baseline_id = left.selectbox("当前基准（Baseline）", list(run_map), index=max(0, len(run_map) - 2), key="eval-baseline")
            candidate_id = right.selectbox("候选版本（Candidate）", list(run_map), index=len(run_map) - 1, key="eval-candidate")
            comparison = lab.compare(baseline_id, candidate_id)
            metric_names = {"accuracy": "总体准确率", "auto_accuracy": "自动准确率", "coverage": "覆盖率", "route_accuracy": "分流准确率", "policy_accuracy": "政策标签准确率", "false_approve_rate": "误放率", "false_reject_rate": "误杀率", "review_rate": "人工复核率"}
            st.dataframe([{"指标": metric_names.get(item.metric, item.metric), "Baseline": f"{item.baseline:.1%}", "Candidate": f"{item.candidate:.1%}", "变化": f"{item.delta:+.1%}"} for item in comparison.metric_deltas], width="stretch", hide_index=True)
            telemetry = st.columns(3)
            telemetry[0].metric("Token 变化", comparison.token_delta)
            telemetry[1].metric("成本变化", f"{comparison.cost_delta:+.4f}")
            telemetry[2].metric("平均延迟变化", f"{comparison.latency_delta_ms:+.1f} ms")
            if comparison.newly_failed_gates:
                st.error("候选版本新增失败门禁：" + "；".join(comparison.newly_failed_gates))
            if comparison.resolved_failed_gates:
                st.success("候选版本已修复：" + "；".join(comparison.resolved_failed_gates))


def render_chunk_preview(chunks: List[PolicyChunk]) -> None:
    st.subheader("策略分块清单")
    policy_ids = sorted({chunk.policy_id for chunk in chunks})
    selected = st.multiselect("策略筛选", policy_ids, default=policy_ids)
    visible = [chunk for chunk in chunks if chunk.policy_id in selected]
    st.dataframe(
        chunk_rows(visible),
        width="stretch",
        hide_index=True,
        column_config={
            "内容预览": st.column_config.TextColumn(width="large"),
            "字符数": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_policy_hub(store: PolicyVectorStore, chunks: List[PolicyChunk], documents: KnowledgeDocumentStore) -> None:
    render_page_header(
        "资产与设置",
        "政策知识库",
        "维护政策原文，检查自动分块，并验证写入 Policy RAG 后的召回结果。",
        [
            (f"{len({item.policy_id for item in chunks})} 个策略组", ""),
            (f"{len(chunks)} 个知识分块", ""),
        ],
    )
    source_tab, upload_tab, search_tab, chunk_tab = st.tabs(["原文库", "上传与处理", "RAG 检索", "知识分块"])
    with source_tab:
        st.markdown("**内置合成政策原文**")
        st.caption(f"`{DEFAULT_POLICY_PATH.name}` · 该文档为公开 Demo 独立编写，不包含公司政策原文。")
        with st.expander("查看完整原文"):
            st.text_area("政策原文", DEFAULT_POLICY_PATH.read_text(encoding="utf-8"), height=420, disabled=True, label_visibility="collapsed")
        uploaded_documents = documents.list(KnowledgeKind.POLICY)
        st.markdown("**上传文档记录**")
        if uploaded_documents:
            st.dataframe([{"文档 ID": item.document_id, "文件名": item.name, "状态": {"stored": "已保存原文", "indexed": "已写入 RAG", "needs_structure": "待整理"}.get(item.status, item.status), "分块数": item.chunk_count, "SHA-256": item.sha256[:16] + "…", "上传时间": item.created_at} for item in uploaded_documents], width="stretch", hide_index=True)
            selected_document = st.selectbox("查看上传原文", uploaded_documents, format_func=lambda item: f"{item.name} · {item.document_id}")
            with st.expander("查看选中文档原文"):
                st.text_area("上传原文", documents.read_text(selected_document), height=320, disabled=True, label_visibility="collapsed")
        else:
            st.info("还没有上传政策文档。")
    with upload_tab:
        st.markdown("**1. 保存原文 → 2. 自动拆分 → 3. 预览 → 4. 写入索引 → 5. 检索验证**")
        uploaded = st.file_uploader("上传政策文档", type=["md", "txt"], key="policy-kb-upload")
        confirmed = st.checkbox("我确认该文档为公开、合成或本人可公开使用的内容，不包含公司机密。")
        if st.button("保存并解析文档", type="primary", disabled=uploaded is None or not confirmed):
            try:
                document = documents.save(uploaded.name, uploaded.getvalue(), KnowledgeKind.POLICY)
                preview_chunks = documents.policy_chunks(document)
                st.session_state["policy-upload-preview"] = (document.document_id, preview_chunks)
                st.success(f"原文已保存，并拆分为 {len(preview_chunks)} 个候选知识分块。")
            except ValueError as exc:
                st.error(str(exc))
        preview = st.session_state.get("policy-upload-preview")
        if preview:
            document_id, preview_chunks = preview
            st.markdown("**候选分块预览**")
            st.dataframe(chunk_rows(preview_chunks), width="stretch", hide_index=True)
            if st.button("确认写入 Policy RAG"):
                document = next((item for item in documents.list(KnowledgeKind.POLICY) if item.document_id == document_id), None)
                if document is None:
                    st.error("找不到待写入的原文记录。")
                else:
                    store.index(preview_chunks, replace=False)
                    documents.mark_indexed(document, len(preview_chunks))
                    st.session_state.pop("policy-upload-preview", None)
                    st.success("已写入 Policy RAG。现在可以到“RAG 检索”验证召回。")
                    st.rerun()
    with search_tab:
        st.caption("输入审核问题，检查系统召回了哪些政策片段及其相似度。召回结果是证据候选，不等于最终违规结论。")
        render_search(store)
    with chunk_tab:
        render_chunk_preview(chunks)


def main() -> None:
    store = get_store()
    brand_library = get_brand_library()
    brand_store = get_brand_store()
    case_store = get_case_store()
    run_store = get_phase9_run_store()
    reviewer_prompt_store = get_reviewer_prompt_store()
    reviewer_prompt_store.ensure_defaults()
    workflow_release_store = get_workflow_release_store()
    knowledge_document_store = get_knowledge_document_store()
    bad_case_store = get_bad_case_store()
    model_configuration_store = get_model_configuration_store()
    review_store = get_review_store()
    staging_store = get_staging_store()
    trace_store = get_trace_store()
    ensure_default_index(store)
    chunks = store.list_chunks()
    llm_settings = runtime_llm_settings()
    prompt_skill_registry = get_prompt_skill_registry()
    workflow_release_store.ensure_defaults(prompt_skill_registry)
    product_release = workflow_release_store.current(WorkflowName.PRODUCT)
    shop_release = workflow_release_store.current(WorkflowName.SHOP)
    product_workflow = ProductWorkflow(
        store,
        brand_store,
        brand_library,
        OfflineFixturePriceProvider(),
        trace_store,
        run_store,
        prompt_version=product_release.prompt_ref if product_release else "PRM-PRODUCT-IPR@v9.1.0",
        skill_version=product_release.skill_ref if product_release else "SKL-PRODUCT-IPR@v9.1.0",
    )
    shop_workflow = ShopWorkflow(
        store,
        brand_store,
        brand_library,
        trace_store,
        run_store,
        prompt_version=shop_release.prompt_ref if shop_release else "PRM-SHOP@v9.1.0",
        skill_version=shop_release.skill_ref if shop_release else "SKL-SHOP@v9.1.0",
    )
    router = ReviewRouter(shop_workflow, product_workflow)
    prompt_experiment_runner = build_reviewer_prompt_runner(llm_settings, store, brand_store, trace_store, run_store)

    with st.sidebar:
        st.markdown(
            """
            <div class="lab-brand">
                <div class="lab-brand__name">Trust & Safety AI Lab</div>
                <div class="lab-brand__meta">决策、评测与人工反馈闭环</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("当前角色")
        role = st.segmented_control(
            "当前角色",
            ["审核员", "策略与 AI 运营"],
            default="审核员",
            key="phase9-role",
            label_visibility="collapsed",
        )
        st.divider()

    reviewer_pages = {
        "审核任务": [
            st.Page(
                lambda: render_phase9_comprehensive_review(router, reviewer_prompt_store, prompt_experiment_runner),
                title="综合审核",
                icon=":material/route:",
                url_path="review",
                default=True,
            ),
            st.Page(
                lambda: render_phase9_shop_review(shop_workflow, reviewer_prompt_store, prompt_experiment_runner),
                title="商家侧审核",
                icon=":material/storefront:",
                url_path="shop-review",
            ),
            st.Page(
                lambda: render_phase9_product_review(product_workflow, reviewer_prompt_store, prompt_experiment_runner),
                title="商品侧审核",
                icon=":material/inventory_2:",
                url_path="product-review",
            ),
        ],
        "我的工作": [
            st.Page(
                lambda: render_phase9_my_prompts(reviewer_prompt_store),
                title="我的 Prompt",
                icon=":material/edit_note:",
                url_path="my-prompts",
            ),
            st.Page(
                lambda: render_phase9_case_list(case_store),
                title="Case 列表",
                icon=":material/list_alt:",
                url_path="cases",
            ),
        ],
    }
    operator_pages = {
        "Workflow 与策略": [
            st.Page(
                lambda: render_phase9_workflow_management(workflow_release_store, prompt_skill_registry),
                title="Workflow 管理",
                icon=":material/account_tree:",
                url_path="workflow-management",
                default=True,
            ),
            st.Page(
                render_prompt_skill_studio,
                title="正式 Prompt 与 Skill",
                icon=":material/tune:",
                url_path="official-prompt-skill",
            ),
            st.Page(
                lambda: render_phase9_reviewer_prompt_observation(reviewer_prompt_store),
                title="审核员 Prompt 观察",
                icon=":material/visibility:",
                url_path="reviewer-prompt-observation",
            ),
        ],
        "RAG 知识库": [
            st.Page(
                lambda: render_policy_hub(store, store.list_chunks(), knowledge_document_store),
                title="PBR 政策知识库",
                icon=":material/library_books:",
                url_path="policies",
            ),
            st.Page(
                lambda: render_phase9_brand_knowledge(brand_store, knowledge_document_store),
                title="品牌商品知识库",
                icon=":material/category:",
                url_path="brand-knowledge",
            ),
        ],
        "质量闭环": [
            st.Page(
                render_evaluation_lab,
                title="Workflow 效果评测",
                icon=":material/monitoring:",
                url_path="workflow-evaluation",
            ),
            st.Page(
                lambda: render_phase9_bad_cases(bad_case_store),
                title="Bad Case 队列",
                icon=":material/error_outline:",
                url_path="bad-cases",
            ),
            st.Page(
                lambda: render_phase9_trace_replay(
                    trace_store,
                    product_workflow,
                    shop_workflow,
                    bad_case_store,
                ),
                title="Trace 与重放",
                icon=":material/history:",
                url_path="traces",
            ),
        ],
        "发布与数据": [
            st.Page(
                lambda: render_system_settings(store),
                title="模型与成本设置",
                icon=":material/settings:",
                url_path="settings",
            ),
            st.Page(
                render_phase9_candidate_dataset,
                title="Candidate Dataset 管理",
                icon=":material/dataset:",
                url_path="candidate-dataset",
            ),
        ],
        "高级实验": [
            st.Page(
                lambda: render_controlled_agent_workspace(
                    store,
                    brand_library,
                    review_store,
                    llm_settings,
                ),
                title="Agent 工具调用实验",
                icon=":material/extension:",
                url_path="agent-tool-lab",
            ),
        ],
    }
    pages = reviewer_pages if role == "审核员" else operator_pages
    page = st.navigation(pages, position="hidden")
    with st.sidebar:
        for section, section_pages in pages.items():
            st.caption(section)
            for navigation_page in section_pages:
                st.page_link(navigation_page, use_container_width=True)
        st.divider()
        if llm_settings.configured:
            st.caption(f"模型就绪 · `{llm_settings.model}`")
        else:
            st.caption("模型未配置 · 离线基线可用")
        st.caption(
            f"政策 {len({chunk.policy_id for chunk in chunks})} 组 · "
            f"知识分块 {len(chunks)} 个"
        )
        st.caption("Prototype · 本地运行")
    page.run()


if __name__ == "__main__":
    main()

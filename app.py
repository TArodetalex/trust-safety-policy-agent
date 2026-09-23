"""Streamlit workspace for policy and product-review workflows."""

from pathlib import Path
import time
from typing import List

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
    DEFAULT_BRAND_LIBRARY_PATH,
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
)
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
    ProductReviewAssistant,
    ProductReviewDecision,
    ProductReviewInput,
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
from trust_safety_agent.trace import TraceStore
from trust_safety_agent.trace_builders import (
    controlled_agent_trace,
    product_review_trace,
    shop_identity_trace,
)
from trust_safety_agent.vector_store import PolicyVectorStore


st.set_page_config(
    page_title="Trust & Safety AI 决策实验室",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    h1 {font-size: 2.4rem !important; line-height: 1.15 !important; letter-spacing: 0;}
    @media (max-width: 640px) {
        h1 {font-size: 2rem !important;}
        .block-container {padding-top: 1.25rem;}
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 12px 16px;
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
    return ControlledBrandLibrary.from_csv(DEFAULT_BRAND_LIBRARY_PATH)


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
    if store.count() == 0:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))


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


def render_sidebar(store: PolicyVectorStore) -> ResilientLLMSettings:
    env_settings = ResilientLLMSettings.from_env()
    with st.sidebar:
        st.header("模型推理")
        api_base = st.text_input(
            "API 地址",
            value=env_settings.api_base,
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
            st.caption("模型状态：已配置")
        else:
            st.caption("模型状态：需要 API Key")

        st.divider()
        st.header("策略知识库")
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

        st.divider()
        st.caption(f"嵌入模型：`{store.embedder.name}`")
        st.caption(f"向量集合：`{store.collection_name}`")
    return ResilientLLMSettings(
        api_key=api_key,
        api_base=api_base,
        model=model,
        timeout_seconds=env_settings.timeout_seconds,
        response_format=response_format,
        fallback_models=tuple(
            item.strip() for item in fallback_models.split(",") if item.strip()
        ),
    )


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
    st.caption(
        f"推理引擎：`{engine}` · 图像证据：`{'是' if image_used else '否'}`"
    )
    status = {
        AgentDecisionLabel.REJECT: "拒绝",
        AgentDecisionLabel.APPROVE: "通过",
        AgentDecisionLabel.NEED_REVIEW: "人工复核",
    }[decision.decision]
    if decision.decision == AgentDecisionLabel.REJECT:
        st.error(f"审核结果：{status}")
    elif decision.decision == AgentDecisionLabel.APPROVE:
        st.success(f"审核结果：{status}")
    else:
        st.warning(f"审核结果：{status}")

    decision_cols = st.columns(3)
    decision_cols[0].metric("置信度", f"{decision.confidence:.0%}")
    decision_cols[1].metric(
        "策略类型",
        decision.policy_label.value if decision.policy_label else "—",
    )
    decision_cols[2].metric("证据数量", len(decision.matched_policy))

    st.markdown(f"**判断理由**  \n{decision.reason}")
    st.markdown(f"**建议操作**  \n{decision.recommended_action}")

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
    st.subheader("商品知识产权审核")
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
    st.subheader("店铺身份审核")
    with st.form("shop-identity"):
        id_col, brand_col = st.columns(2)
        with id_col:
            case_id = st.text_input("案例 ID", value="SHOP-001", key="shop_case_id")
        with brand_col:
            controlled_brand = st.text_input("受控品牌", value="Gucci")
        shop_name = st.text_input("店铺名称", value="Gucci Official Store")
        avatar_url = st.text_input("头像 URL（可选）")
        authorization = st.selectbox(
            "授权状态",
            options=list(AuthorizationStatus),
            index=list(AuthorizationStatus).index(AuthorizationStatus.UNKNOWN),
            format_func=lambda value: {
                AuthorizationStatus.AUTHORIZED: "已授权",
                AuthorizationStatus.UNAUTHORIZED: "未授权",
                AuthorizationStatus.UNKNOWN: "未知",
            }[value],
        )
        avatar_marks = st.text_input(
            "头像视觉标记（逗号分隔）",
            placeholder="Gucci logo, 官方字样",
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
    st.subheader("批量审核")
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
    st.subheader("人工审核工作台")
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
    st.subheader("执行 Trace 与失败分析")
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
    st.subheader("Prompt 与 Skill 工作台")
    st.caption(
        "Prompt 定义模型判断指令；Skill 将 Prompt、工具权限、政策范围、步数和输出契约封装为可复用版本。"
    )
    registry = get_prompt_skill_registry()
    overview_tab, prompt_tab, skill_tab = st.tabs(
        ["版本总览", "新建 Prompt", "新建 Skill"]
    )

    with overview_tab:
        st.markdown("**Prompt 版本**")
        prompts = registry.prompts()
        st.dataframe(
            [
                {
                    "Prompt ID": item.prompt_id,
                    "版本": item.version,
                    "名称": item.name,
                    "工作流": item.workflow.value,
                    "变更说明": item.change_note,
                }
                for item in prompts
            ],
            width="stretch",
            hide_index=True,
        )
        selected_prompt = st.selectbox(
            "查看 Prompt 内容",
            prompts,
            format_func=lambda item: f"{item.name} · {item.prompt_id}@{item.version}",
        )
        st.code(selected_prompt.instructions, language="text")

        st.markdown("**Skill 版本**")
        skills = registry.skills()
        st.dataframe(
            [
                {
                    "Skill ID": item.skill_id,
                    "版本": item.version,
                    "名称": item.name,
                    "工作流": item.workflow.value,
                    "Prompt": f"{item.prompt_id}@{item.prompt_version}",
                    "工具": ", ".join(item.allowed_tools) or "无",
                    "政策范围": ", ".join(item.policy_scope) or "全部",
                    "最大步骤": item.max_steps,
                }
                for item in skills
            ],
            width="stretch",
            hide_index=True,
        )

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
                    "先区分事实证据与推断。判断山寨时，需要同时确认仿制表达或高度近似信号，"
                    "以及受控品牌指向；仅出现品牌名不能直接拒绝。证据冲突时转人工。"
                ),
                height=200,
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
                st.success("Prompt 版本已保存，并可在对应工作流中选择。")
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
                st.success("Skill 版本已保存，并可在对应工作流中选择。")
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_evaluation_lab() -> None:
    st.subheader("AI Evaluation Lab")
    lab = EvaluationLabStore(DEFAULT_EVALUATION_RUNS_PATH)
    with st.expander("新建可复现评测 Run"):
        run_id = st.text_input(
            "Run ID",
            value=f"EV-LAB-{time.strftime('%Y%m%d-%H%M%S')}",
            key="evaluation-lab-run-id",
        )
        st.caption("当前界面运行离线规则基准；真实模型 Run 使用同一产物格式，可在下方直接比较。")
        if st.button(":material/play_arrow: 运行离线规则评测"):
            try:
                with st.spinner("正在评测 Golden Set v4..."):
                    report = run_offline_rules_experiment(
                        run_id=run_id,
                        dataset_path=DEFAULT_GOLDEN_SET_V4_PATH,
                        output_root=DEFAULT_EVALUATION_RUNS_PATH,
                        store=get_store(),
                    )
                st.success(
                    f"{report.eval_run_id} 已完成，"
                    f"自动准确率 {report.metrics.auto_accuracy:.1%}，"
                    f"覆盖率 {report.metrics.coverage:.1%}。"
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    runs = lab.list_runs()
    if not runs:
        st.info("当前没有可比较的评测 Run。")
        return
    run_map = {run.run_id: run for run in runs}
    selected_id = st.selectbox("查看评测 Run", list(run_map), index=len(run_map) - 1)
    selected = run_map[selected_id]
    report = selected.report
    overview = st.columns(5)
    overview[0].metric("自动准确率", f"{report.metrics.auto_accuracy:.1%}")
    overview[1].metric("覆盖率", f"{report.metrics.coverage:.1%}")
    overview[2].metric("分流准确率", f"{report.metrics.route_accuracy:.1%}")
    overview[3].metric("人工复核率", f"{report.metrics.review_rate:.1%}")
    overview[4].metric("质量门禁", "通过" if report.passed else "未通过")
    st.caption(
        f"Engine: `{report.engine}` · Prompt: `{report.prompt_version}` · "
        f"Dataset: `{report.dataset_version}` · Retrieval: `{report.retrieval_version}`"
    )
    if report.failed_gates:
        st.warning("；".join(report.failed_gates))
    if report.metrics.failure_counts:
        st.markdown("**Failure Taxonomy**")
        st.dataframe(
            [
                {"Failure Type": name, "数量": count}
                for name, count in sorted(report.metrics.failure_counts.items())
                if count
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.markdown("**版本对比**")
    if len(runs) < 2:
        st.info("至少需要两个评测 Run 才能进行版本对比。")
        return
    left, right = st.columns(2)
    baseline_id = left.selectbox(
        "Baseline",
        list(run_map),
        index=max(0, len(run_map) - 2),
        key="eval-baseline",
    )
    candidate_id = right.selectbox(
        "Candidate",
        list(run_map),
        index=len(run_map) - 1,
        key="eval-candidate",
    )
    comparison = lab.compare(baseline_id, candidate_id)
    st.dataframe(
        [
            {
                "指标": item.metric,
                "Baseline": f"{item.baseline:.1%}",
                "Candidate": f"{item.candidate:.1%}",
                "变化": f"{item.delta:+.1%}",
            }
            for item in comparison.metric_deltas
        ],
        width="stretch",
        hide_index=True,
    )
    telemetry = st.columns(3)
    telemetry[0].metric("Token 变化", comparison.token_delta)
    telemetry[1].metric("成本变化", f"{comparison.cost_delta:+.4f}")
    telemetry[2].metric("平均延迟变化", f"{comparison.latency_delta_ms:+.1f} ms")
    if comparison.newly_failed_gates:
        st.error("新增失败门禁：" + "；".join(comparison.newly_failed_gates))
    if comparison.resolved_failed_gates:
        st.success("已修复门禁：" + "；".join(comparison.resolved_failed_gates))


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


def main() -> None:
    store = get_store()
    brand_library = get_brand_library()
    review_store = get_review_store()
    staging_store = get_staging_store()
    trace_store = get_trace_store()
    ensure_default_index(store)
    chunks = store.list_chunks()
    llm_settings = render_sidebar(store)

    st.title("Trust & Safety AI 决策实验室")
    st.caption("受控 Agent、政策 RAG、模型评测与人工反馈闭环")

    (
        decision_tab,
        product_tab,
        shop_tab,
        batch_tab,
        reviewer_tab,
        agent_tab,
        prompt_skill_tab,
        evaluation_tab,
        trace_tab,
        search_tab,
        chunks_tab,
    ) = st.tabs(
        [
            "案例审核",
            "商品知识产权",
            "店铺身份",
            "批量审核",
            "人工审核工作台",
            "受控 Agent",
            "Prompt 与 Skill",
            "评测实验室",
            "Trace 与失败分析",
            "策略检索",
            "知识分块",
        ]
    )
    with decision_tab:
        render_case_workspace(store, llm_settings)
    with product_tab:
        render_product_review(store, brand_library, review_store, trace_store)
    with shop_tab:
        render_shop_identity(store, brand_library, review_store, trace_store)
    with batch_tab:
        render_batch_workspace(store, brand_library, review_store, trace_store)
    with reviewer_tab:
        render_reviewer_workspace(review_store, staging_store)
    with agent_tab:
        render_controlled_agent_workspace(
            store, brand_library, review_store, llm_settings
        )
    with prompt_skill_tab:
        render_prompt_skill_studio()
    with evaluation_tab:
        render_evaluation_lab()
    with trace_tab:
        render_trace_workspace(trace_store)
    with search_tab:
        render_search(store)
    with chunks_tab:
        render_chunk_preview(chunks)

    st.divider()
    policy_count = len({chunk.policy_id for chunk in chunks})
    source_count = len({chunk.source for chunk in chunks})
    metric_cols = st.columns(4)
    metric_cols[0].metric("策略组", policy_count)
    metric_cols[1].metric("知识分块", len(chunks))
    metric_cols[2].metric("策略来源", source_count)
    metric_cols[3].metric("嵌入模型", store.embedder.name)


if __name__ == "__main__":
    main()

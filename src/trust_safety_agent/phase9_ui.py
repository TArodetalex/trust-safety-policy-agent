"""Chinese-first Streamlit surfaces for the Phase 9 product views."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from trust_safety_agent.bad_cases import BadCaseRecord, BadCaseSource, BadCaseStore, FailureType, compare_replay
from trust_safety_agent.brand_knowledge import BrandKnowledgeStore, load_brand_knowledge
from trust_safety_agent.case_store import CaseStore, JsonlStore, ProductCase, RunDecision, ShopCase, import_product_cases, import_shop_cases
from trust_safety_agent.config import DEFAULT_BRAND_KNOWLEDGE_PATH, DEFAULT_PHASE9_PRODUCT_DEMOS_PATH, DEFAULT_PHASE9_SHOP_DEMOS_PATH, PROJECT_ROOT
from trust_safety_agent.demo_cases import load_demo_products, load_demo_shops
from trust_safety_agent.model_experiments import WorkflowConfiguration, WorkflowConfigurationStore
from trust_safety_agent.phase9_workflow import ProductRuntimeEvidence, ProductWorkflow, ReviewRouter, ShopRuntimeEvidence, ShopWorkflow
from trust_safety_agent.prompt_skills import PromptSkillRegistry, PromptWorkflow
from trust_safety_agent.reviewer_prompts import ReviewerPromptStore, ReviewerPromptVersion, ReviewerPromptWorkflow
from trust_safety_agent.trace import TraceStore
from trust_safety_agent.workflow_releases import WorkflowName, WorkflowReleaseStore
from trust_safety_agent.knowledge_documents import KnowledgeDocumentStore, KnowledgeKind


def _header(title: str, description: str, eyebrow: str = "审核员") -> None:
    st.markdown(
        f'<div class="lab-page-header"><div class="lab-eyebrow">{eyebrow}</div><div class="lab-page-title">{title}</div><div class="lab-page-desc">{description}</div></div>',
        unsafe_allow_html=True,
    )


def _decision(value: RunDecision) -> None:
    names = {RunDecision.REJECT: "建议拒绝", RunDecision.APPROVE: "建议通过", RunDecision.MANUAL_REVIEW: "转人工复核"}
    css = {RunDecision.REJECT: "reject", RunDecision.APPROVE: "approve", RunDecision.MANUAL_REVIEW: "review"}[value]
    st.markdown(f'<div class="lab-result lab-result--{css}"><div class="lab-result__label">Agent Suggestion</div><div class="lab-result__value">{names[value]}</div></div>', unsafe_allow_html=True)


def _show_product(result) -> None:
    _decision(result.suggested_decision)
    cols = st.columns(4)
    labels = {"hit": "命中", "not_hit": "未命中", "insufficient_evidence": "证据不足", "not_applicable": "不适用"}
    for col, (name, assessment) in zip(cols, result.labels.items()):
        with col:
            st.markdown(f"**{name}**")
            st.write(labels[assessment.status.value])
            st.caption(assessment.reason)
    if result.notices:
        st.warning("；".join(result.notices))
    with st.expander("查看运行证据"):
        st.write({"检测品牌": result.detected_brand, "管控状态": result.brand_control_status.value, "置信度": result.confidence, "Trace ID": result.trace_id})
        if result.price_evidence:
            st.write(result.price_evidence.model_dump(mode="json"))


def _show_shop(result) -> None:
    _decision(result.suggested_decision)
    left, right = st.columns(2)
    left.metric("店铺名称", {"violation": "违规", "compliant": "合规", "manual_review": "转人工"}[result.shop_name_status.value])
    right.metric("店铺头像", {"violation": "违规", "compliant": "合规", "manual_review": "转人工"}[result.shop_avatar_status.value])
    if result.notices:
        st.warning("；".join(result.notices))
    with st.expander("查看运行证据"):
        st.write({"检测品牌": result.detected_brand, "管控状态": result.brand_control_status.value, "置信度": result.confidence, "Trace ID": result.trace_id})


def _demo_maps():
    shops = {case.shop_id: (case, evidence, scenario) for case, evidence, scenario in load_demo_shops(DEFAULT_PHASE9_SHOP_DEMOS_PATH)}
    products = {case.product_id: (case, evidence, scenario) for case, evidence, scenario in load_demo_products(DEFAULT_PHASE9_PRODUCT_DEMOS_PATH)}
    return shops, products


def _analysis_mode(
    store: ReviewerPromptStore,
    workflow: ReviewerPromptWorkflow,
    *,
    key: str,
) -> tuple[str, Optional[ReviewerPromptVersion]]:
    mode = st.segmented_control(
        "分析模式",
        ["正式 Workflow", "我的 Prompt"],
        default="正式 Workflow",
        key=f"{key}-analysis-mode",
    )
    if mode == "正式 Workflow":
        return mode, None
    records = [item for item in store.list() if item.owner_type.value == "reviewer" and item.workflow in {workflow, ReviewerPromptWorkflow.COMPREHENSIVE}]
    if not records:
        st.warning("当前场景还没有可用的自定义 Prompt。请先到“我的 Prompt”创建并保存版本。")
        return mode, None
    selected = st.selectbox(
        "Prompt 版本",
        records,
        format_func=lambda item: f"{item.name} · {item.prompt_id}@v{item.version}",
        key=f"{key}-prompt-version",
    )
    st.caption(
        f"实验运行 · Policy RAG {'开启' if selected.use_policy_rag else '关闭'} · "
        f"Brand RAG {'开启' if selected.use_brand_rag else '关闭'} · 不覆盖正式结果"
    )
    return mode, selected


def _show_experimental(result: Any, prompt: ReviewerPromptVersion) -> None:
    st.info(f"实验结果 · {prompt.name} · {prompt.prompt_id}@v{prompt.version}")
    st.write({
        "决策": getattr(getattr(result, "decision", None), "value", getattr(result, "decision", "unknown")),
        "置信度": getattr(result, "confidence", None),
        "原因": getattr(result, "reason", ""),
        "建议动作": getattr(result, "recommended_action", ""),
        "Trace ID": getattr(result, "trace_id", ""),
    })
    st.caption("该结果仅用于比较 Prompt 效果，不写入正式 Agent Suggestion。")


def render_comprehensive_review(
    router: ReviewRouter,
    prompt_store: ReviewerPromptStore,
    experiment_runner: Optional[Callable[[Any, ReviewerPromptVersion], Any]] = None,
) -> None:
    _header("综合审核", "选择已有 Case，或同时输入 Shop 与 Product 字段。系统会拆成两个独立子任务并分别保留结果。")
    shops, products = _demo_maps()
    analysis_mode, prompt = _analysis_mode(prompt_store, ReviewerPromptWorkflow.COMPREHENSIVE, key="comprehensive")
    mode = st.segmented_control("输入方式", ["从 Case 列表选择", "手动输入"], default="从 Case 列表选择")
    if mode == "从 Case 列表选择":
        kind = st.radio("Case 类型", ["Shop Case", "Product Case"], horizontal=True)
        options = list(shops) if kind == "Shop Case" else list(products)
        selected = st.selectbox("选择 Case", options)
        record = shops[selected] if kind == "Shop Case" else products[selected]
        st.caption(f"场景：`{record[2]}`")
        button_label = "运行正式 Workflow" if analysis_mode == "正式 Workflow" else "运行 Prompt 实验"
        if st.button(button_label, type="primary", use_container_width=True, disabled=analysis_mode == "我的 Prompt" and prompt is None):
            with st.spinner("正在执行可观测 Workflow..."):
                if analysis_mode == "我的 Prompt":
                    if experiment_runner is None:
                        st.error("模型尚未配置，Prompt 已保存但无法执行在线实验。")
                    else:
                        _show_experimental(experiment_runner(record[0], prompt), prompt)
                else:
                    session = router.run(shop_case=record[0], shop_evidence=record[1]) if kind == "Shop Case" else router.run(product_case=record[0], product_evidence=record[1])
                    _show_shop(session.shop_result) if session.shop_result else _show_product(session.product_result)
        return

    include_shop = st.checkbox("包含 Shop 子任务")
    include_product = st.checkbox("包含 Product 子任务")
    shop_case = None
    product_case = None
    if include_shop:
        left, right = st.columns(2)
        shop_id = left.text_input("Shop ID", key="manual-shop-id")
        shop_name = right.text_input("店铺名称", key="manual-shop-name")
        avatar = left.text_input("头像 URL 或本地路径", key="manual-shop-avatar")
        brand = right.text_input("商家填报 Brand", key="manual-shop-brand") or None
        shop_case = ShopCase(shop_id=shop_id, shop_name=shop_name, shop_avatar=avatar, brand=brand) if shop_id and shop_name and avatar else None
    if include_product:
        left, right = st.columns(2)
        product_id = left.text_input("Product ID", key="manual-product-id")
        title = right.text_input("商品标题", key="manual-product-title")
        image = left.text_input("商品图片 URL 或本地路径", key="manual-product-image")
        brand = right.text_input("商品填报 Brand", key="manual-product-brand") or None
        product_case = ProductCase(product_id=product_id, title=title, product_images=[image], brand=brand) if product_id and title and image else None
    if st.button("拆分并运行", type="primary", disabled=shop_case is None and product_case is None):
        if analysis_mode == "我的 Prompt":
            if prompt is None or experiment_runner is None:
                st.error("请选择 Prompt 版本并配置可用模型。")
                return
            if shop_case:
                st.subheader("Shop 实验子任务")
                _show_experimental(experiment_runner(shop_case, prompt), prompt)
            if product_case:
                st.subheader("Product 实验子任务")
                _show_experimental(experiment_runner(product_case, prompt), prompt)
            return
        session = router.run(shop_case=shop_case, product_case=product_case)
        if session.shop_result:
            st.subheader("Shop 子任务")
            _show_shop(session.shop_result)
        if session.product_result:
            st.subheader("Product 子任务")
            _show_product(session.product_result)


def render_shop_review(workflow: ShopWorkflow, prompt_store: ReviewerPromptStore, experiment_runner: Optional[Callable[[Any, ReviewerPromptVersion], Any]] = None) -> None:
    _header("商家侧审核", "分别判断 Shop Name 与 Shop Avatar；授权只豁免原始 Brand 对应的品牌。")
    shops, _ = _demo_maps()
    analysis_mode, prompt = _analysis_mode(prompt_store, ReviewerPromptWorkflow.SHOP, key="shop")
    selected = st.selectbox("Shop Case", list(shops), format_func=lambda key: f"{key} · {shops[key][0].shop_name}")
    case, evidence, scenario = shops[selected]
    left, right = st.columns([1, 1.3])
    with left:
        image = PROJECT_ROOT / case.shop_avatar
        if image.exists():
            st.image(str(image), caption=case.shop_name, use_container_width=True)
    with right:
        st.write({"Shop ID": case.shop_id, "店铺名称": case.shop_name, "原始 Brand": case.brand, "授权状态": case.brand_authorized, "演示场景": scenario})
        button_label = "运行商家侧 Workflow" if analysis_mode == "正式 Workflow" else "运行 Prompt 实验"
        if st.button(button_label, type="primary", use_container_width=True, disabled=analysis_mode == "我的 Prompt" and prompt is None):
            if analysis_mode == "我的 Prompt":
                if experiment_runner is None:
                    st.error("模型尚未配置，暂时不能执行 Prompt 实验。")
                else:
                    st.session_state["phase9-shop-experiment"] = (experiment_runner(case, prompt), prompt)
            else:
                st.session_state["phase9-shop-result"] = workflow.run(case, evidence)[0]
    if analysis_mode == "正式 Workflow" and st.session_state.get("phase9-shop-result"):
        _show_shop(st.session_state["phase9-shop-result"])
    if analysis_mode == "我的 Prompt" and st.session_state.get("phase9-shop-experiment"):
        _show_experimental(*st.session_state["phase9-shop-experiment"])


def render_product_review(workflow: ProductWorkflow, prompt_store: ReviewerPromptStore, experiment_runner: Optional[Callable[[Any, ReviewerPromptVersion], Any]] = None) -> None:
    _header("商品侧审核", "同时分析 Counterfeit、Knockoff、MBA 和 TMI，不强行选择主标签。")
    _, products = _demo_maps()
    analysis_mode, prompt = _analysis_mode(prompt_store, ReviewerPromptWorkflow.PRODUCT, key="product")
    selected = st.selectbox("Product Case", list(products), format_func=lambda key: f"{key} · {products[key][0].title}")
    case, evidence, scenario = products[selected]
    left, right = st.columns([1, 1.3])
    with left:
        image = PROJECT_ROOT / case.product_images[0]
        if image.exists():
            st.image(str(image), caption=case.title, use_container_width=True)
    with right:
        st.write({"Product ID": case.product_id, "标题": case.title, "描述": case.description, "原始 Brand": case.brand, "授权状态": case.brand_authorized, "价格": case.price, "演示场景": scenario})
        button_label = "运行商品侧 Workflow" if analysis_mode == "正式 Workflow" else "运行 Prompt 实验"
        if st.button(button_label, type="primary", use_container_width=True, disabled=analysis_mode == "我的 Prompt" and prompt is None):
            if analysis_mode == "我的 Prompt":
                if experiment_runner is None:
                    st.error("模型尚未配置，暂时不能执行 Prompt 实验。")
                else:
                    st.session_state["phase9-product-experiment"] = (experiment_runner(case, prompt), prompt)
            else:
                st.session_state["phase9-product-result"] = workflow.run(case, evidence)[0]
    if analysis_mode == "正式 Workflow" and st.session_state.get("phase9-product-result"):
        _show_product(st.session_state["phase9-product-result"])
    if analysis_mode == "我的 Prompt" and st.session_state.get("phase9-product-experiment"):
        _show_experimental(*st.session_state["phase9-product-experiment"])


def render_my_prompts(store: ReviewerPromptStore) -> None:
    _header("我的 Prompt", "自定义分析仅产生 Experimental 结果，不覆盖正式 Agent Suggestion。")
    create_tab, version_tab = st.tabs(["新建 Prompt", "新建版本"])
    with create_tab:
        with st.form("reviewer-create-prompt"):
            name = st.text_input("名称", placeholder="例如：商品 Logo 证据专项分析")
            workflow = st.selectbox("适用场景", list(ReviewerPromptWorkflow), format_func=lambda item: {ReviewerPromptWorkflow.COMPREHENSIVE: "综合审核", ReviewerPromptWorkflow.SHOP: "商家侧审核", ReviewerPromptWorkflow.PRODUCT: "商品侧审核"}[item])
            instructions = st.text_area("分析指令", height=160, placeholder="至少 20 个字符。说明要关注的证据、判断边界与转人工条件。")
            col1, col2 = st.columns(2)
            policy_rag = col1.checkbox("使用 PBR 政策 RAG", value=True)
            brand_rag = col2.checkbox("使用品牌商品 RAG", value=True)
            save = st.form_submit_button("保存 Prompt v1", type="primary")
        if save:
            try:
                store.save(ReviewerPromptVersion(name=name.strip(), workflow=workflow, instructions=instructions.strip(), use_policy_rag=policy_rag, use_brand_rag=brand_rag))
                st.success("Prompt v1 已保存，可在对应审核页面选择。")
                st.rerun()
            except ValueError as exc:
                st.error(f"无法保存：{exc}")
    with version_tab:
        existing = store.list()
        if not existing:
            st.info("请先创建一个 Prompt。")
        else:
            with st.form("reviewer-new-version"):
                base = st.selectbox("基于已有版本", existing, format_func=lambda item: f"{item.name} · {item.prompt_id}@v{item.version}")
                revised = st.text_area("新版本指令", value=base.instructions, height=160)
                new_policy = st.checkbox("使用 PBR 政策 RAG", value=base.use_policy_rag, key="new-version-policy")
                new_brand = st.checkbox("使用品牌商品 RAG", value=base.use_brand_rag, key="new-version-brand")
                save_version = st.form_submit_button("保存为新版本", type="primary")
            if save_version:
                try:
                    saved = store.new_version(base.prompt_id, revised.strip(), use_policy_rag=new_policy, use_brand_rag=new_brand)
                    st.success(f"已保存 {saved.prompt_id}@v{saved.version}。")
                    st.rerun()
                except ValueError as exc:
                    st.error(f"无法保存：{exc}")
    records = store.list()
    if not records:
        st.info("还没有自定义 Prompt。")
        return
    st.dataframe([{"Prompt ID": item.prompt_id, "版本": item.version, "名称": item.name, "场景": item.workflow.value, "Policy RAG": item.use_policy_rag, "Brand RAG": item.use_brand_rag, "状态": item.status.value} for item in records], width="stretch", hide_index=True)
    st.caption("实验运行保留输入校验、统一 Schema、敏感信息过滤与独立 Trace，不覆盖正式决策。")


def _csv_bytes(records: List[dict]) -> bytes:
    if not records:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8-sig")


def render_case_list(store: CaseStore) -> None:
    _header("Case 列表", "查看、筛选、导出、创建修订或归档。归档不会物理删除历史记录。")
    shops, products = _demo_maps()
    if not store.list_shops(include_archived=True) and not store.list_products(include_archived=True):
        for case, _, _ in shops.values():
            store.create_shop(case)
        for case, _, _ in products.values():
            store.create_product(case)
    add_tab, import_tab = st.tabs(["直接新增", "批量导入"])
    with add_tab:
        shop_form, product_form = st.tabs(["新增 Shop Case", "新增 Product Case"])
        with shop_form:
            with st.form("add-shop-case", clear_on_submit=True):
                shop_id = st.text_input("Shop ID")
                shop_name = st.text_input("店铺名称")
                shop_avatar = st.text_input("头像 URL 或项目内路径")
                shop_brand = st.text_input("Brand（可选）")
                shop_auth = st.selectbox("授权状态", ["未知", "已授权", "未授权"])
                add_shop = st.form_submit_button("添加 Shop Case", type="primary")
            if add_shop:
                try:
                    auth = {"未知": None, "已授权": True, "未授权": False}[shop_auth]
                    store.create_shop(ShopCase(shop_id=shop_id.strip(), shop_name=shop_name.strip(), shop_avatar=shop_avatar.strip(), brand=shop_brand.strip() or None, brand_authorized=auth))
                    st.success("Shop Case 已添加。")
                    st.rerun()
                except ValueError as exc:
                    st.error(f"无法添加：{exc}")
        with product_form:
            with st.form("add-product-case", clear_on_submit=True):
                product_id = st.text_input("Product ID")
                title = st.text_input("商品标题")
                description = st.text_area("商品描述")
                image_refs = st.text_area("商品图片 URL 或项目内路径（每行一个）")
                first, second = st.columns(2)
                price = first.number_input("价格（0 表示未知）", min_value=0.0, value=0.0)
                currency = second.text_input("币种", value="USD", max_chars=3)
                product_brand = first.text_input("Brand（可选）")
                product_auth = second.selectbox("授权状态", ["未知", "已授权", "未授权"])
                add_product = st.form_submit_button("添加 Product Case", type="primary")
            if add_product:
                try:
                    auth = {"未知": None, "已授权": True, "未授权": False}[product_auth]
                    images = [item.strip() for item in image_refs.splitlines() if item.strip()]
                    store.create_product(ProductCase(product_id=product_id.strip(), title=title.strip(), description=description.strip(), product_images=images, price=price or None, currency=currency.strip().upper() if price else None, brand=product_brand.strip() or None, brand_authorized=auth))
                    st.success("Product Case 已添加。")
                    st.rerun()
                except ValueError as exc:
                    st.error(f"无法添加：{exc}")
    with import_tab:
        import_kind = st.radio("导入类型", ["Shop Cases", "Product Cases"], horizontal=True)
        uploaded = st.file_uploader("选择文件", type=["csv", "xlsx"], key="phase9-case-import")
        if uploaded and st.button("校验并导入"):
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(uploaded.getvalue())
                temporary_path = Path(temporary.name)
            try:
                imported = import_shop_cases(temporary_path, store) if import_kind == "Shop Cases" else import_product_cases(temporary_path, store)
                st.success(f"已导入 {len(imported)} 条 Case。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            finally:
                temporary_path.unlink(missing_ok=True)
    shop_tab, product_tab = st.tabs(["Shop Cases", "Product Cases"])
    with shop_tab:
        records = store.list_shops(include_archived=True)
        rows = [item.model_dump(mode="json") for item in records]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.download_button("导出 Shop CSV", _csv_bytes(rows), "shop_cases.csv", "text/csv")
    with product_tab:
        records = store.list_products(include_archived=True)
        rows = [item.model_dump(mode="json") for item in records]
        for row in rows:
            row["product_images"] = " | ".join(row["product_images"])
        st.dataframe(rows, width="stretch", hide_index=True)
        st.download_button("导出 Product CSV", _csv_bytes(rows), "product_cases.csv", "text/csv")
    st.caption("直接新增与 CSV/Excel 导入使用同一 Case Store；历史修订采用追加记录，不做物理删除。")


def render_workflow_management(store: WorkflowReleaseStore, registry: PromptSkillRegistry) -> None:
    _header("Workflow 管理", "正式资产需要显式发布；保存 Skill 不会自动影响审核端。发布与回退都会保留历史。", "策略与 AI 运营")
    store.ensure_defaults(registry)
    names = {WorkflowName.ROUTER: "综合审核路由", WorkflowName.SHOP: "商家侧审核", WorkflowName.PRODUCT: "商品侧审核"}
    current = [store.current(item) for item in WorkflowName]
    st.dataframe([{"Workflow": names[item.workflow], "配置版本": f"v{item.configuration_version}", "Skill": item.skill_ref, "Prompt": item.prompt_ref, "状态": "Published", "发布时间": item.created_at} for item in current if item], width="stretch", hide_index=True)
    publish_tab, history_tab = st.tabs(["发布配置", "发布历史"])
    with publish_tab:
        workflow = st.selectbox("目标 Workflow", list(WorkflowName), format_func=lambda item: names[item])
        allowed_workflow = PromptWorkflow.CONTROLLED_AGENT if workflow == WorkflowName.PRODUCT else PromptWorkflow.GENERAL_CASE
        skills = registry.skills(allowed_workflow)
        skill = st.selectbox("选择 Skill 版本", skills, format_func=lambda item: f"{item.name} · {item.skill_id}@{item.version}")
        resolved = registry.resolve_skill(skill.skill_id, skill.version)
        st.caption(f"绑定 Prompt：`{resolved.prompt.prompt_id}@{resolved.prompt.version}`")
        note = st.text_input("发布说明", placeholder="例如：Candidate 评测通过，发布到商品审核")
        first, second = st.columns(2)
        if first.button("发布此配置", type="primary", use_container_width=True):
            release = store.publish(workflow, skill.skill_id, skill.version, registry, note=note)
            st.success(f"已发布配置 v{release.configuration_version}，审核端后续运行将使用新版本。")
            st.rerun()
        if second.button("回退到上一配置", use_container_width=True):
            try:
                release = store.rollback(workflow, registry)
                st.success(f"已创建回退发布 v{release.configuration_version}。")
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
    with history_tab:
        history = sorted(store.list(), key=lambda item: item.created_at, reverse=True)
        st.dataframe([{"Workflow": names[item.workflow], "配置版本": item.configuration_version, "Skill": item.skill_ref, "Prompt": item.prompt_ref, "前一发布": item.previous_release_id or "-", "说明": item.note, "时间": item.created_at} for item in history], width="stretch", hide_index=True)
    st.code("validate_input → normalize_and_extract_evidence → recall_entities → retrieve_policy → retrieve_brand_knowledge → evaluate_rules → aggregate_decision → apply_guardrail → publish_result", language=None)


def render_brand_knowledge(store: BrandKnowledgeStore, documents: KnowledgeDocumentStore) -> None:
    _header("品牌商品知识库", "保留品牌知识原文，并将品牌实体、别名、品类和代表产品作为独立 Brand RAG。", "策略与 AI 运营")
    chunks = load_brand_knowledge(DEFAULT_BRAND_KNOWLEDGE_PATH)
    source_tab, search_tab, upload_tab = st.tabs(["原文库", "RAG 检索", "上传参考文档"])
    with source_tab:
        st.caption(f"内置原文：`{DEFAULT_BRAND_KNOWLEDGE_PATH.name}` · 50 个公共品牌，来源与歧义说明均保留在原文中。")
        with st.expander("查看品牌知识库完整原文"):
            st.text_area("品牌知识原文", DEFAULT_BRAND_KNOWLEDGE_PATH.read_text(encoding="utf-8"), height=420, disabled=True, label_visibility="collapsed")
        uploaded = documents.list(KnowledgeKind.BRAND)
        if uploaded:
            st.dataframe([{"文档 ID": item.document_id, "文件名": item.name, "状态": "已保存原文，待结构化校验", "SHA-256": item.sha256[:16] + "…"} for item in uploaded], width="stretch", hide_index=True)
    with search_tab:
        query = st.text_input("检索品牌、中文别名或代表产品", value="耐克 跑鞋 Air Force")
        if query:
            hits = store.retrieve(query, top_k=8)
            category_names = {
                "consumer electronics": "消费电子",
                "consumer electronics and home appliances": "消费电子与家电",
                "consumer electronics and gaming": "消费电子与游戏",
                "gaming hardware and software": "游戏硬件与软件",
                "sportswear and footwear": "运动服饰与鞋类",
                "athletic apparel": "运动服饰",
                "beauty and skincare": "美妆与护肤",
                "beauty, skincare, and haircare": "美妆、护肤与护发",
                "construction toys": "积木与拼装玩具",
                "cosmetics": "彩妆",
                "denim and apparel": "牛仔与服饰",
                "dolls and toys": "玩偶与玩具",
                "drinkware and outdoor gear": "饮具与户外装备",
                "eyewear": "眼镜",
                "fashion and accessories": "时尚与配饰",
                "fashion and apparel": "时尚服装",
                "footwear": "鞋类",
                "footwear and accessories": "鞋类与配饰",
                "footwear and apparel": "鞋类与服饰",
                "handbags and accessories": "手袋与配饰",
                "home appliances and personal care": "家电与个人护理",
                "jewelry and accessories": "珠宝与配饰",
                "jewelry and watches": "珠宝与腕表",
                "kitchen appliances": "厨房电器",
                "luxury fashion and beauty": "奢侈时尚与美妆",
                "luxury fashion and leather goods": "奢侈时尚与皮具",
                "luxury fashion, fragrance, and watches": "奢侈时尚、香水与腕表",
                "luxury watches": "奢侈腕表",
                "outdoor apparel and equipment": "户外服饰与装备",
                "performance eyewear and apparel": "专业眼镜与运动服饰",
                "skincare": "护肤",
                "skincare and personal care": "护肤与个人护理",
            }
            rows = []
            for hit in hits:
                chinese_alias = next((alias for alias in hit.chunk.aliases if any("\u4e00" <= char <= "\u9fff" for char in alias)), "")
                display_name = f"{hit.chunk.brand_name}（{chinese_alias}）" if chinese_alias else hit.chunk.brand_name
                rows.append({"Brand ID": hit.chunk.chunk_id, "品牌": display_name, "品类": category_names.get(hit.chunk.category, hit.chunk.category), "代表产品": "；".join(hit.chunk.representative_products), "普通词/歧义说明": hit.chunk.ambiguity_note, "相似度": round(hit.score, 3)})
            st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("品牌召回只提供实体与商品上下文，不会因为召回到某个品牌就直接判定违规。")
    with upload_tab:
        st.info("品牌知识需要标准字段：品牌名、别名、品类、代表产品、歧义说明、检索词和公开来源。任意原文会先保存，校验通过后才可进入 Brand RAG。")
        brand_file = st.file_uploader("上传品牌参考文档", type=["md", "txt"], key="brand-kb-upload")
        brand_confirmed = st.checkbox("我确认该文档来自公开信息，不包含内部品牌管控名单。")
        if st.button("保存品牌原文", disabled=brand_file is None or not brand_confirmed):
            try:
                document = documents.save(brand_file.name, brand_file.getvalue(), KnowledgeKind.BRAND)
                st.success(f"已保存 {document.name}。当前作为原文资产保留，尚未写入 Brand RAG。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.caption(f"索引版本 v1.0.0 · 公共管控品牌 {len(chunks)} 个 · Collection `{store.collection_name}`")


def render_reviewer_prompt_observation(store: ReviewerPromptStore) -> None:
    _header("审核员 Prompt 观察", "查看审核员实验版本；复制为 Candidate 后仍需经过评测才能发布。", "策略与 AI 运营")
    records = store.list()
    st.dataframe([item.model_dump(mode="json") for item in records], width="stretch", hide_index=True) if records else st.info("暂无审核员 Prompt。")
    if records:
        selected = st.selectbox("选择版本", [f"{item.prompt_id}@{item.version}" for item in records])
        if st.button("复制为 Candidate Prompt"):
            prompt_id, version = selected.split("@")
            candidate = store.copy_to_candidate(prompt_id, int(version))
            st.success(f"已创建 {candidate.prompt_id}@{candidate.version}")


def render_bad_cases(store: BadCaseStore) -> None:
    _header("Bad Case 队列", "集中处理审核结果有误、证据不足、评测失败或规则冲突的 Case；系统只建议原因，最终归因由人工确认。", "策略与 AI 运营")
    st.markdown("**处理动线**　① Case 自动/手动入队　→　② 查看原结果与证据　→　③ 检查 Trace　→　④ 重放验证假设　→　⑤ 人工确认根因　→　⑥ 进入 Prompt、RAG 或 Workflow 优化")
    with st.expander("什么是失败原因分类（Failure Taxonomy）？"):
        st.write("它不是新的审核政策，而是给 AI 系统错误分组：问题究竟出在输入、证据提取、RAG 召回、Prompt、模型推理、输出格式、聚合规则还是人工标注分歧。")
        st.dataframe([
            {"系统代码": "input_missing / evidence_extraction_error", "中文含义": "输入缺失或证据提取错误", "常见动作": "补字段、OCR 或图片质量校验"},
            {"系统代码": "policy_retrieval_miss / brand_retrieval_miss", "中文含义": "RAG 没召回正确政策或品牌", "常见动作": "调整分块、查询或知识原文"},
            {"系统代码": "prompt_instruction_error / model_reasoning_error", "中文含义": "Prompt 指令或模型推理错误", "常见动作": "改 Prompt、Few-shot 或候选模型"},
            {"系统代码": "schema_error / aggregation_error / guardrail_error", "中文含义": "结构化输出、结果聚合或安全门禁错误", "常见动作": "修 Schema 或 Workflow 代码"},
            {"系统代码": "annotation_disagreement / unknown", "中文含义": "人工分歧或暂时无法归因", "常见动作": "策略运营复核并补充说明"},
        ], width="stretch", hide_index=True)
    records = store.list()
    if not records:
        st.info("当前没有 Bad Case。低置信度、人工分歧、规则冲突和评测失败会进入这里。")
        return
    st.dataframe([item.model_dump(mode="json") for item in records], width="stretch", hide_index=True)
    pending = [item for item in records if item.confirmed_failure_type is None]
    if pending:
        selected = st.selectbox("待确认记录", [item.bad_case_id for item in pending])
        failure = st.selectbox("确认根因", [item.value for item in FailureType])
        note = st.text_area("归因说明")
        if st.button("确认归因"):
            store.confirm(selected, FailureType(failure), "strategy-ops", note)
            st.success("最终根因已由人工确认。")
            st.rerun()


def render_trace_replay(trace_store: TraceStore, product_workflow: ProductWorkflow, shop_workflow: ShopWorkflow, bad_store: BadCaseStore) -> None:
    _header("Trace 与重放", "Trace 回答“这次结果是怎么一步步得到的”；重放用于只改变一个条件，再运行一次验证错误原因。", "策略与 AI 运营")
    st.markdown("**操作动线**　① 选择历史 Run　→　② 从上到下检查节点　→　③ 提出一个归因假设　→　④ 只修正一个变量并重放　→　⑤ 查看首次出现差异的节点")
    with st.expander("九个节点分别在做什么？"):
        st.dataframe([
            {"节点": "validate_input", "含义": "检查必填字段与图片是否存在"},
            {"节点": "normalize_and_extract_evidence", "含义": "清洗文本并整理图片、Logo、授权等证据"},
            {"节点": "recall_entities", "含义": "识别可能涉及的品牌和商品"},
            {"节点": "retrieve_policy", "含义": "从 Policy RAG 召回适用规则"},
            {"节点": "retrieve_brand_knowledge", "含义": "从 Brand RAG 获取品牌别名和商品上下文"},
            {"节点": "evaluate_rules", "含义": "逐项判断 Counterfeit、Knockoff、MBA、TMI 或店铺身份"},
            {"节点": "aggregate_decision", "含义": "把多标签结果汇总为通过、拒绝或转人工建议"},
            {"节点": "apply_guardrail", "含义": "拦截冲突、低置信度和不合法输出"},
            {"节点": "publish_result", "含义": "保存正式结果、版本与 Trace"},
        ], width="stretch", hide_index=True)
    traces = [item for item in trace_store.list_traces() if item.run_id]
    if not traces:
        st.info("先在审核端运行一条 Phase 9 Case，随后可在这里重放。")
        return
    selected_id = st.selectbox("选择历史运行（Run）", [item.run_id for item in reversed(traces)])
    original = next(item for item in traces if item.run_id == selected_id)
    node_names = {"validate_input": "输入校验", "normalize_and_extract_evidence": "证据整理", "recall_entities": "品牌/商品识别", "retrieve_policy": "Policy RAG 召回", "retrieve_brand_knowledge": "Brand RAG 召回", "retrieve_reference_price": "参考价检索", "evaluate_rules": "逐项规则判断", "aggregate_decision": "结果聚合", "apply_guardrail": "安全门禁", "publish_result": "保存结果", "model_inference": "模型推理", "publish_experimental_result": "保存实验结果"}
    st.dataframe([{"步骤": index, "节点": f"{node_names.get(item.node_name, item.node_name)}（{item.node_name}）", "状态": item.status.value, "耗时 ms": item.latency_ms, "输出摘要": str(item.output)[:280]} for index, item in enumerate(original.nodes, 1)], width="stretch", hide_index=True)
    shops, products = _demo_maps()
    if original.case_id not in shops and original.case_id not in products:
        st.warning("该历史运行没有可重放的 Phase 9 合成 Case 快照。")
        return
    st.markdown("**重放假设：品牌识别可能有误**")
    corrected_brand = st.text_input("本次只修正识别品牌", value="", placeholder="例如：Nike")
    if st.button("创建 Replay Run", type="primary", disabled=not corrected_brand.strip()):
        if original.case_id in products:
            case, evidence, _ = products[original.case_id]
            replay_trace = product_workflow.run(case, evidence.model_copy(update={"detected_brand": corrected_brand}), parent_run_id=original.run_id)[2]
        else:
            case, evidence, _ = shops[original.case_id]
            replay_trace = shop_workflow.run(case, evidence.model_copy(update={"detected_name_brand": corrected_brand}), parent_run_id=original.run_id)[2]
        comparison = compare_replay(original, replay_trace)
        st.session_state["phase9-replay"] = (comparison, replay_trace)
    replay_state = st.session_state.get("phase9-replay")
    if replay_state:
        comparison, replay_trace = replay_state
        left, right = st.columns(2)
        left.markdown("**原运行**")
        left.write(original.final_output)
        right.markdown("**重放运行**")
        right.write(replay_trace.final_output)
        st.success(f"首次差异节点：{comparison.first_differing_node or '无差异'}；建议归因：{comparison.suggested_failure_type.value}；归因状态：待确认")
        if st.button("加入 Bad Case 队列"):
            bad_store.enqueue(BadCaseRecord(run_id=original.run_id or "unknown", case_id=original.case_id, source=BadCaseSource.RESULT_REPORTED, suspected_failure_type=comparison.suggested_failure_type))
            st.success("已加入 Bad Case 队列，等待策略运营确认根因。")


def render_model_release(store: WorkflowConfigurationStore) -> None:
    _header("模型实验与发布", "固定 Dataset、Workflow、Prompt、Skill、索引与参数，只改变模型 ID。", "策略与 AI 运营")
    workflow = st.selectbox("Workflow", ["product_ipr", "shop_identity"])
    model = st.text_input("候选模型 ID")
    if st.button("创建 Draft", disabled=not model.strip()):
        config = WorkflowConfiguration(workflow=workflow, model_id=model, dataset_version="phase9-candidate-v1", workflow_version="v9.1.0", prompt_version="official-v9.1.0", skill_version="official-v9.1.0", policy_index_version="synthetic-policy-v2", brand_index_version="v1.0.0")
        store.save(config)
        st.success(f"已创建 {config.config_id}")
    records = store.store.list()
    if records:
        st.dataframe([item.model_dump(mode="json") for item in records], width="stretch", hide_index=True)


def render_candidate_dataset() -> None:
    _header("Candidate Dataset 管理", "新 Case 与人工反馈进入候选集，不直接污染 Golden Set v4。", "策略与 AI 运营")
    shops, products = _demo_maps()
    cols = st.columns(3)
    cols[0].metric("Shop Candidate", len(shops))
    cols[1].metric("Product Candidate", len(products))
    cols[2].metric("Golden Set v4", "冻结")
    st.dataframe([{"Case ID": key, "类型": "Shop", "场景": value[2]} for key, value in shops.items()] + [{"Case ID": key, "类型": "Product", "场景": value[2]} for key, value in products.items()], width="stretch", hide_index=True)

"""Streamlit workspace for the Policy KB milestone."""

from typing import List

import streamlit as st

from trust_safety_agent.adjudicator import PolicyAdjudicator
from trust_safety_agent.config import DEFAULT_CHROMA_DIRECTORY, DEFAULT_POLICY_PATH
from trust_safety_agent.llm_adjudicator import LLMPolicyAdjudicator
from trust_safety_agent.llm_client import (
    ImageInput,
    LLMSettings,
    OpenAICompatibleClient,
)
from trust_safety_agent.policy_loader import load_policy_file, load_policy_text
from trust_safety_agent.schema import (
    AgentDecision,
    AgentDecisionLabel,
    ContentType,
    PolicyChunk,
)
from trust_safety_agent.vector_store import PolicyVectorStore


st.set_page_config(
    page_title="Trust & Safety Policy Agent",
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


def ensure_default_index(store: PolicyVectorStore) -> None:
    if store.count() == 0:
        store.index(load_policy_file(DEFAULT_POLICY_PATH))


def chunk_rows(chunks: List[PolicyChunk]) -> List[dict]:
    return [
        {
            "policy_id": chunk.policy_id,
            "title": chunk.title,
            "label": chunk.policy_label.value if chunk.policy_label else "",
            "exemption": (
                "" if chunk.exemption_type.value == "none" else chunk.exemption_type.value
            ),
            "chunk_id": chunk.chunk_id,
            "characters": len(chunk.content),
            "preview": chunk.content[:180].replace("\n", " "),
        }
        for chunk in chunks
    ]


def render_sidebar(store: PolicyVectorStore) -> LLMSettings:
    env_settings = LLMSettings.from_env()
    with st.sidebar:
        st.header("Inference")
        api_base = st.text_input(
            "API base",
            value=env_settings.api_base,
            key="llm_api_base",
        )
        model = st.text_input(
            "Vision model",
            value=env_settings.model,
            key="llm_model",
        )
        api_key = st.text_input(
            "API key",
            value=env_settings.api_key,
            type="password",
            key="llm_api_key",
        )
        if api_key.strip():
            st.caption("LLM status: configured")
        else:
            st.caption("LLM status: API key required")

        st.divider()
        st.header("Policy source")
        uploaded = st.file_uploader("Markdown document", type=["md", "txt"])
        max_chars = st.number_input(
            "Chunk size",
            min_value=300,
            max_value=2000,
            value=900,
            step=100,
        )
        overlap_chars = st.number_input(
            "Overlap",
            min_value=0,
            max_value=400,
            value=120,
            step=20,
        )

        if st.button(
            ":material/database: Build index",
            type="primary",
            use_container_width=True,
        ):
            if overlap_chars >= max_chars:
                st.error("Overlap must be smaller than chunk size.")
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
                st.toast(f"Indexed {len(chunks)} policy chunks.")
                st.rerun()

        st.divider()
        st.caption(f"Embedding: `{store.embedder.name}`")
        st.caption(f"Collection: `{store.collection_name}`")
    return LLMSettings(
        api_key=api_key,
        api_base=api_base,
        model=model,
        timeout_seconds=env_settings.timeout_seconds,
    )


def render_search(store: PolicyVectorStore) -> None:
    st.subheader("Policy retrieval")
    query_col, k_col = st.columns([5, 1])
    with query_col:
        query = st.text_input(
            "Case or policy query",
            value="Gucci replica handbag sold as a 1:1 copy",
        )
    with k_col:
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)

    if query.strip():
        for hit in store.retrieve(query, top_k=int(top_k)):
            label = (
                hit.chunk.policy_label.value
                if hit.chunk.policy_label
                else hit.chunk.exemption_type.value
            )
            with st.expander(
                f"#{hit.rank} · {hit.chunk.policy_id} · {hit.chunk.title} "
                f"· score {hit.score:.3f}",
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
        f"Engine: `{engine}` · Image evidence: `{'yes' if image_used else 'no'}`"
    )
    status = decision.decision.value.replace("_", " ").upper()
    if decision.decision == AgentDecisionLabel.REJECT:
        st.error(f"Decision: {status}")
    elif decision.decision == AgentDecisionLabel.APPROVE:
        st.success(f"Decision: {status}")
    else:
        st.warning(f"Decision: {status}")

    decision_cols = st.columns(3)
    decision_cols[0].metric("Confidence", f"{decision.confidence:.0%}")
    decision_cols[1].metric(
        "Policy",
        decision.policy_label.value if decision.policy_label else "—",
    )
    decision_cols[2].metric("Evidence", len(decision.matched_policy))

    st.markdown(f"**Reason**  \n{decision.reason}")
    st.markdown(f"**Recommended action**  \n{decision.recommended_action}")

    if decision.matched_policy:
        st.markdown("**Policy evidence**")
        for evidence in decision.matched_policy:
            with st.expander(
                f"{evidence.policy_id} · {evidence.chunk_id}",
                expanded=True,
            ):
                if evidence.retrieval_score is not None:
                    st.caption(f"Retrieval score: {evidence.retrieval_score:.3f}")
                st.markdown(f"> {evidence.quote}")

    with st.expander("Structured output"):
        st.code(decision.model_dump_json(indent=2), language="json")


def render_case_workspace(
    store: PolicyVectorStore,
    llm_settings: LLMSettings,
) -> None:
    st.subheader("Case adjudication")
    with st.form("case-adjudication"):
        engine = st.segmented_control(
            "Inference engine",
            options=["Multimodal LLM", "Offline rules"],
            default="Multimodal LLM",
        )
        content_type = st.selectbox(
            "Content type",
            options=list(ContentType),
            index=list(ContentType).index(ContentType.PRODUCT),
            format_func=lambda value: value.value.replace("_", " ").title(),
        )
        input_text = st.text_area(
            "Content",
            value="Gucci 1:1 mirror copy handbag, includes branded dust bag",
            height=120,
            max_chars=2000,
        )
        image_col, url_col = st.columns(2)
        with image_col:
            uploaded_image = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="case_image",
            )
        with url_col:
            image_url = st.text_input(
                "Image URL",
                placeholder="https://example.com/product.jpg",
            )
        if uploaded_image:
            st.image(uploaded_image, width=280)
        submitted = st.form_submit_button(
            ":material/gavel: Run decision",
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

            if engine == "Offline rules":
                decision = PolicyAdjudicator(store).adjudicate(
                    case_id="C000",
                    content_type=content_type,
                    input_text=input_text,
                )
                engine_name = "offline-rules-v1"
            else:
                if not llm_settings.configured:
                    raise ValueError("Configure an API key and vision model first.")
                client = OpenAICompatibleClient(llm_settings)
                with st.spinner(f"Running {client.model}..."):
                    decision = LLMPolicyAdjudicator(store, client).adjudicate(
                        case_id="C000",
                        content_type=content_type,
                        input_text=input_text,
                        images=images,
                    )
                engine_name = client.model
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


def render_chunk_preview(chunks: List[PolicyChunk]) -> None:
    st.subheader("Chunk inventory")
    policy_ids = sorted({chunk.policy_id for chunk in chunks})
    selected = st.multiselect("Policy filter", policy_ids, default=policy_ids)
    visible = [chunk for chunk in chunks if chunk.policy_id in selected]
    st.dataframe(
        chunk_rows(visible),
        use_container_width=True,
        hide_index=True,
        column_config={
            "preview": st.column_config.TextColumn(width="large"),
            "characters": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    store = get_store()
    ensure_default_index(store)
    chunks = store.list_chunks()
    llm_settings = render_sidebar(store)

    st.title("Trust & Safety Policy Agent")
    st.caption("Multimodal policy-grounded adjudication")

    decision_tab, search_tab, chunks_tab = st.tabs(
        ["Case decision", "Retrieval", "Chunks"]
    )
    with decision_tab:
        render_case_workspace(store, llm_settings)
    with search_tab:
        render_search(store)
    with chunks_tab:
        render_chunk_preview(chunks)

    st.divider()
    policy_count = len({chunk.policy_id for chunk in chunks})
    source_count = len({chunk.source for chunk in chunks})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Policy groups", policy_count)
    metric_cols[1].metric("Chunks", len(chunks))
    metric_cols[2].metric("Sources", source_count)
    metric_cols[3].metric("Embedding", store.embedder.name)


if __name__ == "__main__":
    main()

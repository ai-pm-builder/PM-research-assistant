from __future__ import annotations

import textwrap

import streamlit as st

from config import settings
from db import init_db, list_briefs_with_reports
from research_pipeline import ingest_documents, run_research


def init_app() -> None:
    """One-time app initialization."""
    init_db()


def render_setup_page() -> None:
    st.header("Setup & Data")
    st.write(
        "Upload your internal docs and customer feedback. "
        "We'll index them into SQLite and Chroma so they can power future research."
    )
    docs_files = st.file_uploader(
        "Upload internal docs (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    csv_files = st.file_uploader(
        "Upload customer feedback / reviews (CSV)",
        type=["csv"],
        accept_multiple_files=True,
    )
    product_area = st.text_input("Optional product area tag", "")

    if st.button("Index documents"):
        if not docs_files and not csv_files:
            st.warning("Please upload at least one document or CSV file.")
            return

        with st.spinner("Indexing documents into SQLite + Chroma..."):
            summary = ingest_documents(
                uploaded_docs=docs_files,
                uploaded_csvs=csv_files,
                product_area=product_area or None,
            )

        st.success(
            f"Ingestion complete. Indexed {summary.documents_count} documents "
            f"into {summary.chunks_count} text chunks."
        )


def render_new_research_page() -> None:
    st.header("New Research")
    st.write(
        "Describe your product area and feature idea. "
        "We'll run a research pass across internal docs, feedback, and competitors."
    )
    brief_text = st.text_area("Describe the product area and feature idea", height=200)
    target_users = st.text_input("Target users")
    goal_metric = st.text_input("Primary goal/metric")
    competitors_raw = st.text_input("Competitors (comma-separated domains)")

    with st.expander("Advanced settings"):
        n_internal = st.slider("Internal doc chunks to retrieve", min_value=5, max_value=50, value=20, step=5)
        n_feedback = st.slider("Feedback/review chunks to retrieve", min_value=10, max_value=80, value=40, step=10)
        store_comp = st.checkbox("Store fetched competitor pages into Chroma for reuse", value=False)

    if st.button("Run research"):
        if not brief_text.strip():
            st.warning("Please provide a brief description of the product area and feature idea.")
            return

        with st.spinner("Running research pipeline (this may take a minute)..."):
            try:
                report_md = run_research(
                    brief_text=brief_text.strip(),
                    target_users=target_users.strip(),
                    goal_metric=goal_metric.strip(),
                    competitors_raw=competitors_raw.strip(),
                    n_internal=n_internal,
                    n_feedback=n_feedback,
                    store_competitors_in_rag=store_comp,
                )
            except Exception as exc:
                st.error(f"Research failed: {exc}")
                return

        st.subheader("Research report")
        st.markdown(report_md, unsafe_allow_html=False)


def render_past_research_page() -> None:
    st.header("Past Research")
    st.write("Revisit previous research briefs and reports.")

    rows = list_briefs_with_reports()
    if not rows:
        st.info("No research briefs found yet. Run a new research brief first.")
        return

    st.subheader("Briefs")
    table_data = [
        {
            "ID": row["id"],
            "Created at": row["created_at"],
            "Title": row["title"],
            "Product": row["product"],
            "Feature": row["feature_name"],
            "Status": row["status"],
        }
        for row in rows
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    options = {f"{row['id']} – {row['title']} (status: {row['status']})": row for row in rows}
    label = st.selectbox("Select a brief to view details", list(options.keys()))
    selected = options[label]

    st.subheader("Brief summary")
    st.write(
        textwrap.dedent(
            f"""
            - **Product**: {selected['product'] or 'n/a'}
            - **Feature**: {selected['feature_name'] or 'n/a'}
            - **Status**: {selected['status']}
            - **Created at**: {selected['created_at']}
            """
        )
    )

    report_md = selected["report_markdown"]
    if report_md:
        st.subheader("Report")
        st.markdown(report_md, unsafe_allow_html=False)
    else:
        st.info("No report has been generated for this brief yet.")


def main() -> None:
    st.set_page_config(
        page_title="PM Research Assistant",
        layout="wide",
    )

    init_app()

    with st.sidebar:
        st.title("PM Research Assistant")
        st.caption("Gemini + Chroma + SQLite")
        page = st.radio(
            "Navigation",
            options=["Setup & Data", "New Research", "Past Research"],
        )

        if not settings.gemini_api_key:
            st.error(
                "GEMINI_API_KEY is not set. Add it to a .env file or your environment "
                "so the research features can run."
            )

    if page == "Setup & Data":
        render_setup_page()
    elif page == "New Research":
        render_new_research_page()
    else:
        render_past_research_page()


if __name__ == "__main__":
    main()


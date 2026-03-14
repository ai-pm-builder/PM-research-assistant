from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from config import settings
from db import (
    insert_document,
    insert_research_brief,
    insert_research_report,
    update_research_brief_fields,
    update_research_brief_status,
)
from llm_client import generate_research_report, normalize_brief
from rag_store import add_documents, query_documents
from scraper import extract_text_from_file, fetch_url_text


@dataclass
class IngestionSummary:
    documents_count: int
    chunks_count: int


def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Simple sliding-window character chunking."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def ingest_documents(
    uploaded_docs: Sequence[Any],
    uploaded_csvs: Sequence[Any],
    product_area: Optional[str] = None,
) -> IngestionSummary:
    """
    Ingest uploaded documents and CSVs into SQLite + Chroma.

    `uploaded_docs` and `uploaded_csvs` are expected to be Streamlit UploadedFile-like
    objects (with .name and .read()) but are typed as Any to avoid a hard dependency.
    """
    documents_count = 0
    chunks_count = 0

    # Handle text-like documents.
    for up in uploaded_docs or []:
        file_name = getattr(up, "name", "document")
        raw_bytes = up.read()
        dest_path = settings.uploads_dir / file_name
        dest_path.write_bytes(raw_bytes)

        text = extract_text_from_file(dest_path)
        file_chunks = _chunk_text(text)

        doc_id = insert_document(
            doc_type="internal_doc",
            title=file_name,
            source=file_name,
            product_area=product_area,
            raw_path=str(dest_path),
        )
        documents_count += 1

        ids = [f"{doc_id}:{i}" for i in range(len(file_chunks))]
        metadatas = [
            {
                "doc_type": "internal_doc",
                "source": file_name,
                "product_area": product_area,
            }
            for _ in file_chunks
        ]
        add_documents(ids=ids, texts=file_chunks, metadatas=metadatas)
        chunks_count += len(file_chunks)

    # Handle CSV-based feedback.
    for up in uploaded_csvs or []:
        file_name = getattr(up, "name", "feedback.csv")
        raw_bytes = up.read()
        dest_path = settings.uploads_dir / file_name
        dest_path.write_bytes(raw_bytes)

        df = pd.read_csv(BytesIO(raw_bytes))

        # Guess doc_type from common columns; fallback to support_feedback.
        doc_type = "support_feedback"
        if any(col.lower().startswith("rating") for col in df.columns):
            doc_type = "public_review"

        # Store one document row per CSV file.
        doc_id = insert_document(
            doc_type=doc_type,
            title=file_name,
            source=file_name,
            product_area=product_area,
            raw_path=str(dest_path),
        )
        documents_count += 1

        row_texts: List[str] = []
        for _, row in df.iterrows():
            parts: List[str] = []
            if "title" in df.columns:
                parts.append(f"Title: {row.get('title', '')}")
            if "comment" in df.columns:
                parts.append(f"Comment: {row.get('comment', '')}")
            if "rating" in df.columns:
                parts.append(f"Rating: {row.get('rating', '')}")
            if "platform" in df.columns:
                parts.append(f"Platform: {row.get('platform', '')}")
            if "competitor_name" in df.columns:
                parts.append(f"Competitor: {row.get('competitor_name', '')}")

            if not parts:
                # Fallback: dump the entire row as text.
                parts = [f"{col}: {row[col]}" for col in df.columns]

            row_texts.append("\n".join(str(p) for p in parts))

        # Chunk by grouping multiple rows together into larger texts.
        combined_text = "\n\n---\n\n".join(row_texts)
        feedback_chunks = _chunk_text(combined_text)

        ids = [f"{doc_id}:{i}" for i in range(len(feedback_chunks))]
        metadatas = [
            {
                "doc_type": doc_type,
                "source": file_name,
                "product_area": product_area,
            }
            for _ in feedback_chunks
        ]
        add_documents(ids=ids, texts=feedback_chunks, metadatas=metadatas)
        chunks_count += len(feedback_chunks)

    return IngestionSummary(documents_count=documents_count, chunks_count=chunks_count)


def run_research(
    brief_text: str,
    target_users: str,
    goal_metric: str,
    competitors_raw: str,
    n_internal: int = 20,
    n_feedback: int = 40,
    store_competitors_in_rag: bool = False,
) -> str:
    """
    Run the full research pipeline for a new brief and return the Markdown report.
    """
    # 1) Create research brief record in DB with status pending -> running.
    title = (brief_text[:60] + "...") if len(brief_text) > 60 else brief_text
    brief_id = insert_research_brief(
        title=title or "New research brief",
        raw_brief=brief_text,
        product=None,
        feature_name=None,
        user_segments=target_users or None,
        goal_metric=goal_metric or None,
        competitors=competitors_raw or None,
        status="pending",
    )
    update_research_brief_status(brief_id, "running")

    try:
        # 2) Normalize brief via LLM.
        normalized = normalize_brief(brief_text)
        # Enrich with UI fields if missing.
        if target_users and not normalized.get("user_segments"):
            normalized["user_segments"] = target_users
        if goal_metric and not normalized.get("goal_metric"):
            normalized["goal_metric"] = goal_metric

        # Persist key normalized fields.
        update_research_brief_fields(
            brief_id=brief_id,
            product=normalized.get("product") or None,
            feature_name=normalized.get("feature_name") or None,
            user_segments=normalized.get("user_segments") or None,
            goal_metric=normalized.get("goal_metric") or None,
            competitors=",".join(normalized.get("competitors", [])) if normalized.get("competitors") else None,
        )

        # 3) Build RAG queries.
        product = normalized.get("product", "")
        feature_name = normalized.get("feature_name", "")
        user_segments_norm = normalized.get("user_segments", "")
        related_topics = normalized.get("related_topics", [])

        queries = [
            " ".join(filter(None, [product, feature_name, user_segments_norm])),
            f"complaints requests pain points about {feature_name or ', '.join(related_topics)}",
        ]

        # Internal docs.
        internal_results = query_documents(
            query_texts=queries,
            n_results=n_internal,
            where={"doc_type": "internal_doc"},
        )
        internal_chunks: List[str] = []
        for docs in internal_results.get("documents", []):
            internal_chunks.extend(docs)

        # Feedback docs.
        feedback_results = query_documents(
            query_texts=queries,
            n_results=n_feedback,
            where={"doc_type": {"$in": ["support_feedback", "public_review"]}},
        )
        feedback_chunks: List[str] = []
        for docs in feedback_results.get("documents", []):
            feedback_chunks.extend(docs)

        # 4) Competitor pages / reviews.
        competitor_domains: List[str] = []
        if competitors_raw:
            competitor_domains = [c.strip() for c in competitors_raw.split(",") if c.strip()]
        elif normalized.get("competitors"):
            competitor_domains = [str(c).strip() for c in normalized["competitors"] if str(c).strip()]

        competitor_texts: List[str] = []
        for domain in competitor_domains:
            if not domain:
                continue
            url = domain
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"https://{domain}"
            text = fetch_url_text(url)
            if text:
                competitor_texts.append(f"Source: {url}\n\n{text}")

        # Optionally store competitor pages in Chroma for reuse.
        if store_competitors_in_rag and competitor_texts:
            ids = []
            metadatas = []
            for idx, text in enumerate(competitor_texts):
                ids.append(f"competitor:{brief_id}:{idx}")
                metadatas.append({"doc_type": "competitor_page", "source": "web", "brief_id": brief_id})
            add_documents(ids=ids, texts=competitor_texts, metadatas=metadatas)

        # 5) Synthesis via Gemini.
        report_markdown = generate_research_report(
            brief=normalized,
            internal_chunks=internal_chunks,
            feedback_chunks=feedback_chunks,
            competitor_chunks=competitor_texts,
        )

        # 6) Persist report and mark brief as done.
        insert_research_report(brief_id=brief_id, report_markdown=report_markdown)
        update_research_brief_status(brief_id, "done")

        return report_markdown

    except Exception:
        update_research_brief_status(brief_id, "error")
        raise



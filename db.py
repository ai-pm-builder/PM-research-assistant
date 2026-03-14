from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from config import settings


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = str(db_path or settings.db_path)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                product_area TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                raw_path TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_briefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                product TEXT,
                feature_name TEXT,
                user_segments TEXT,
                goal_metric TEXT,
                competitors TEXT,
                raw_brief TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief_id INTEGER NOT NULL,
                report_markdown TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (brief_id) REFERENCES research_briefs(id)
            );
            """
        )


def insert_document(
    doc_type: str,
    title: str,
    source: str,
    product_area: Optional[str],
    raw_path: Optional[str],
) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents (type, title, source, product_area, raw_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_type, title, source, product_area, raw_path),
        )
        return int(cur.lastrowid)


def insert_research_brief(
    title: str,
    raw_brief: str,
    product: Optional[str] = None,
    feature_name: Optional[str] = None,
    user_segments: Optional[str] = None,
    goal_metric: Optional[str] = None,
    competitors: Optional[str] = None,
    status: str = "pending",
) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO research_briefs (
                title, product, feature_name, user_segments,
                goal_metric, competitors, raw_brief, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                product,
                feature_name,
                user_segments,
                goal_metric,
                competitors,
                raw_brief,
                status,
            ),
        )
        return int(cur.lastrowid)


def update_research_brief_status(brief_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE research_briefs SET status = ? WHERE id = ?",
            (status, brief_id),
        )


def insert_research_report(brief_id: int, report_markdown: str) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO research_reports (brief_id, report_markdown)
            VALUES (?, ?)
            """,
            (brief_id, report_markdown),
        )
        return int(cur.lastrowid)


def update_research_brief_fields(
    brief_id: int,
    product: Optional[str],
    feature_name: Optional[str],
    user_segments: Optional[str],
    goal_metric: Optional[str],
    competitors: Optional[str],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE research_briefs
            SET product = ?, feature_name = ?, user_segments = ?, goal_metric = ?, competitors = ?
            WHERE id = ?
            """,
            (product, feature_name, user_segments, goal_metric, competitors, brief_id),
        )


def list_briefs_with_reports() -> List[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                rb.id,
                rb.title,
                rb.product,
                rb.feature_name,
                rb.status,
                rb.created_at,
                rr.report_markdown
            FROM research_briefs rb
            LEFT JOIN research_reports rr ON rb.id = rr.brief_id
            ORDER BY rb.created_at DESC
            """
        )
        return cur.fetchall()


def get_report_by_brief_id(brief_id: int) -> Optional[str]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT report_markdown FROM research_reports WHERE brief_id = ? ORDER BY created_at DESC LIMIT 1",
            (brief_id,),
        )
        row = cur.fetchone()
        return row["report_markdown"] if row else None



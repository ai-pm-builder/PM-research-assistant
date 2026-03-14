from __future__ import annotations

from pathlib import Path
from typing import Optional

import pdfplumber
import requests
from bs4 import BeautifulSoup


def fetch_url_text(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return a cleaned text representation."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network issues
        return f"[Error fetching {url}: {exc}]"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

    parts = []
    if title:
        parts.append(f"# {title}")
    if headings:
        parts.append("\n".join(headings))
    if paragraphs:
        parts.append("\n\n".join(paragraphs))

    return "\n\n".join(parts)


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                texts.append(txt)
    return "\n\n".join(texts)


def extract_text_from_file(path: Path) -> str:
    """Extract text from txt, md, or pdf files."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    # Fallback: best-effort read
    return path.read_text(encoding="utf-8", errors="ignore")



from __future__ import annotations

from typing import Any, Dict, Iterable, List

import google.generativeai as genai

from config import settings


_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; please configure it in your environment or .env file.")
    genai.configure(api_key=settings.gemini_api_key)
    _configured = True


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using Gemini embeddings."""
    _ensure_configured()
    if not texts:
        return []

    # Simple loop for now; can be batched/optimized later.
    vectors: List[List[float]] = []
    for text in texts:
        resp = genai.embed_content(
            model=settings.gemini_embed_model,
            content=text,
        )
        vectors.append(resp["embedding"])
    return vectors


def normalize_brief(raw_text: str) -> Dict[str, Any]:
    """Ask Gemini to normalize a free-text research brief into a structured dict."""
    _ensure_configured()
    system_prompt = (
        "You are a product research planner helping a product manager clarify a research brief. "
        "Given the raw description, extract the following fields and respond with ONLY valid JSON:\n"
        "{\n"
        '  "product": string,\n'
        '  "feature_name": string,\n'
        '  "user_segments": string,\n'
        '  "goal_metric": string,\n'
        '  "competitors": string[],\n'
        '  "related_topics": string[]\n'
        "}\n"
        "If something is missing, infer a reasonable placeholder but keep it realistic and concise."
    )

    model = genai.GenerativeModel(settings.gemini_chat_model)
    response = model.generate_content(
        [
            system_prompt,
            f"Raw brief:\n{raw_text}",
        ]
    )
    text = response.text or "{}"

    import json

    # Try to parse JSON directly, with simple fallback heuristics.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring if the model added prose.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    # Ensure all expected keys exist.
    data.setdefault("product", "")
    data.setdefault("feature_name", "")
    data.setdefault("user_segments", "")
    data.setdefault("goal_metric", "")
    data.setdefault("competitors", [])
    data.setdefault("related_topics", [])

    return data


def generate_research_report(
    brief: Dict[str, Any],
    internal_chunks: List[str],
    feedback_chunks: List[str],
    competitor_chunks: List[str],
) -> str:
    """Generate a Markdown research report using Gemini."""
    _ensure_configured()
    model = genai.GenerativeModel(settings.gemini_chat_model)

    product = brief.get("product", "")
    feature_name = brief.get("feature_name", "")
    user_segments = brief.get("user_segments", "")
    goal_metric = brief.get("goal_metric", "")
    competitors = brief.get("competitors", [])

    def _join_snippets(snippets: List[str], max_chars: int = 6000) -> str:
        buf: List[str] = []
        total = 0
        for s in snippets:
            if total + len(s) > max_chars:
                break
            buf.append(s)
            total += len(s)
        return "\n\n---\n\n".join(buf)

    internal_text = _join_snippets(internal_chunks)
    feedback_text = _join_snippets(feedback_chunks)
    competitor_text = _join_snippets(competitor_chunks)

    system_prompt = (
        "You are a senior product researcher. "
        "Using the provided brief and evidence, write a concise, well-structured Markdown report.\n\n"
        "Structure your report with the following sections (use Markdown headings):\n"
        "1. Executive summary\n"
        "2. Problem & user needs\n"
        "3. Competitive landscape (include a Markdown table where possible)\n"
        "4. Customer voice (themes + representative quotes)\n"
        "5. Opportunities & risks\n"
        "6. Open questions & suggested follow-up research\n\n"
        "Be opinionated but grounded in the evidence. When you reference evidence, make it clear which source it is from "
        "(internal docs, customer feedback, or competitor material)."
    )

    user_prompt = (
        f"Brief context:\n"
        f"- Product: {product}\n"
        f"- Feature: {feature_name}\n"
        f"- Target users: {user_segments}\n"
        f"- Primary goal/metric: {goal_metric}\n"
        f"- Competitors: {', '.join(competitors) if competitors else 'n/a'}\n\n"
        "Evidence from internal docs:\n"
        f"{internal_text}\n\n"
        "Evidence from customer feedback (support tickets, reviews, NPS, etc.):\n"
        f"{feedback_text}\n\n"
        "Evidence from competitor pages and reviews:\n"
        f"{competitor_text}\n\n"
        "Now produce the Markdown report."
    )

    response = model.generate_content([system_prompt, user_prompt])
    return response.text or ""



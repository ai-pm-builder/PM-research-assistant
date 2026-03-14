from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from llm_client import embed_texts


_COLLECTION_NAME = "product_research_docs"
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _client


def get_collection():
    """Return (and lazily create) the Chroma collection used for research docs."""
    global _collection
    if _collection is not None:
        return _collection

    client = _get_client()

    # Use a custom embedding function that delegates to Gemini via llm_client.embed_texts
    class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
            return embed_texts(texts)

    embedding_fn = GeminiEmbeddingFunction()

    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def add_documents(
    ids: List[str],
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Add or update documents in the Chroma collection."""
    if not ids:
        return
    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)


def query_documents(
    query_texts: List[str],
    n_results: int = 20,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query the Chroma collection and return raw results."""
    if not query_texts:
        return {"ids": [], "documents": [], "metadatas": []}
    collection = get_collection()
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
    )



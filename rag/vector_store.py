"""
Vector retrieval: ChromaDB (persistent) or in-memory FAISS + optional embeddings.
Controlled by ``AGVL_VECTOR_BACKEND``: ``none`` | ``chroma`` | ``faiss``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag.embeddings import embed_texts


def _kb_docs(kb_dir: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not kb_dir.is_dir():
        return rows
    for path in sorted(kb_dir.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.append((path.name, text))
    return rows


def _chroma_query(
    query: str,
    docs: list[tuple[str, str]],
    persist_root: Path,
    k: int,
) -> list[dict[str, Any]]:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        return []

    if not docs:
        return []

    persist_root.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("AGVL_CHROMA_COLLECTION", "agvl_kb")
    client = chromadb.PersistentClient(path=str(persist_root))
    ef = embedding_functions.DefaultEmbeddingFunction()
    coll = client.get_or_create_collection(name=name, embedding_function=ef)
    ids = [d[0] for d in docs]
    texts = [d[1][:8000] for d in docs]
    coll.upsert(ids=ids, documents=texts)
    res = coll.query(query_texts=[query], n_results=min(k, len(ids)))
    chunks: list[dict[str, Any]] = []
    if not res["documents"] or not res["documents"][0]:
        return chunks
    for i, doc in enumerate(res["documents"][0]):
        sid = res["ids"][0][i] if res.get("ids") else ids[0]
        chunks.append({"source": sid, "text": doc, "vector": "chroma"})
    return chunks


def _faiss_query(query: str, docs: list[tuple[str, str]], k: int) -> list[dict[str, Any]]:
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        return []

    if not docs:
        return []

    texts = [d[1][:8000] for d in docs]
    mat = embed_texts(texts + [query])
    if mat is None or len(mat) < 2:
        return []

    import numpy as np

    doc_vecs = mat[:-1].astype("float32")
    q = mat[-1:].astype("float32")
    faiss.normalize_L2(doc_vecs)
    faiss.normalize_L2(q)
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    scores, idxs = index.search(q, min(k, doc_vecs.shape[0]))
    chunks: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0], strict=False):
        if idx < 0:
            continue
        name, body = docs[int(idx)]
        chunks.append(
            {
                "source": name,
                "text": body[:8000],
                "score": float(score),
                "vector": "faiss",
            }
        )
    return chunks


def vector_chunks(query: str, kb_dir: Path, package_root: Path) -> list[dict[str, Any]]:
    backend = os.environ.get("AGVL_VECTOR_BACKEND", "none").lower().strip()
    docs = _kb_docs(kb_dir)
    k = int(os.environ.get("AGVL_VECTOR_TOP_K", "5"))
    if backend == "chroma":
        persist = package_root / "data" / "chroma"
        return _chroma_query(query, docs, persist, k)
    if backend == "faiss":
        return _faiss_query(query, docs, k)
    return []

"""Combine file baseline, vector (Chroma/FAISS), and graph context for S2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.graph_rag import graph_chunks_for_query
from rag.vector_store import vector_chunks


def _file_chunks(kb_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not kb_dir.is_dir():
        return chunks
    for path in sorted(kb_dir.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append({"source": path.name, "text": text[:8000], "vector": "file"})
    return chunks


def hybrid_retrieve(
    query: str,
    *,
    kb_dir: Path,
    package_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"vector_backend": None, "graph": False}

    vchunks = vector_chunks(query, kb_dir, package_root)
    if vchunks:
        meta["vector_backend"] = vchunks[0].get("vector")

    gchunks = graph_chunks_for_query(query)
    if gchunks:
        meta["graph"] = True

    if vchunks or gchunks:
        merged = vchunks + gchunks
        return merged, meta

    return _file_chunks(kb_dir), meta

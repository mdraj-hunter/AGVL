"""S2: RAGRetriever (ChromaDB + sentence-transformers with keyword fallback)."""

from pathlib import Path

from stages.rag_retriever import RAGRetriever

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _PACKAGE_ROOT / "knowledge_base"


def run(context: dict) -> dict:
    q = str(context.get("validated_input") or context.get("query") or "")
    retriever = RAGRetriever(_KB_DIR, persist_dir=_PACKAGE_ROOT / "data" / "chroma_rag")
    raw_chunks = retriever.retrieve(q, top_k=5)
    chunks: list[dict] = []
    backend = "none"
    for ch in raw_chunks:
        meta = ch.get("metadata") or {}
        backend = str(meta.get("backend", backend))
        chunks.append(
            {
                "text": ch.get("text", ""),
                "source": ch.get("source", meta.get("source", "unknown")),
                "metadata": meta,
            }
        )
    out = dict(context)
    out["retrieved_chunks"] = chunks
    out["rag_meta"] = {"top_k": 5, "backend": backend, "count": len(chunks)}
    return out

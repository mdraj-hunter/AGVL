"""S2: RAGRetriever — ChromaDB + sentence-transformers over knowledge_base/."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


class RAGRetriever:
    """Loads ``knowledge_base`` text files into Chroma; returns top-k chunks with sources."""

    def __init__(
        self,
        kb_dir: Path,
        *,
        collection_name: str = "agvl_kb",
        persist_dir: Path | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._kb_dir = Path(kb_dir)
        root = Path(__file__).resolve().parent.parent
        self._persist = persist_dir or (root / "data" / "chroma_rag")
        self._collection_name = collection_name
        self._embed_model = embed_model or os.environ.get(
            "AGVL_EMBED_MODEL", "all-MiniLM-L6-v2"
        )

    def _load_docs(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        if not self._kb_dir.is_dir():
            return rows
        for path in sorted(self._kb_dir.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rows.append((path.name, text))
        return rows

    def _fallback_chunks(self, query: str, top_k: int) -> list[dict[str, Any]]:
        docs = self._load_docs()
        if not docs:
            return []
        q_tokens = set(re.findall(r"[A-Za-z0-9']+", query.lower()))
        scored: list[tuple[float, str, str]] = []
        for name, body in docs:
            low = body.lower()
            score = sum(1 for t in q_tokens if len(t) > 1 and t in low)
            scored.append((score, name, body))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, name, body in scored[:top_k]:
            out.append(
                {
                    "text": body[:2000],
                    "source": name,
                    "metadata": {"source": name, "score": float(score), "backend": "keyword"},
                }
            )
        return out

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError:
            return self._fallback_chunks(query, top_k)

        docs = self._load_docs()
        if not docs:
            return []

        self._persist.mkdir(parents=True, exist_ok=True)
        ef = SentenceTransformerEmbeddingFunction(model_name=self._embed_model)
        client = chromadb.PersistentClient(path=str(self._persist))
        coll = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=ef,
        )
        ids = [d[0] for d in docs]
        texts = [d[1][:12000] for d in docs]
        coll.upsert(ids=ids, documents=texts, metadatas=[{"source": i} for i in ids])

        res = coll.query(query_texts=[query], n_results=min(top_k, len(ids)))
        chunks: list[dict[str, Any]] = []
        if not res.get("documents") or not res["documents"][0]:
            return self._fallback_chunks(query, top_k)
        for i, doc in enumerate(res["documents"][0]):
            sid = res["ids"][0][i] if res.get("ids") else ids[0]
            dist = None
            if res.get("distances") and res["distances"][0]:
                dist = float(res["distances"][0][i])
            chunks.append(
                {
                    "text": doc,
                    "source": sid,
                    "metadata": {"source": sid, "distance": dist, "backend": "chroma"},
                }
            )
        return chunks

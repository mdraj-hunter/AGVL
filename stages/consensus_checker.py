"""S6: ConsensusChecker — Claude vs OpenAI answer similarity via embedding cosine."""

from __future__ import annotations

import math
import os
import re
from typing import Any

from components.tool_llm import complete_claude, complete_openai


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"[A-Za-z0-9']+", a.lower()))
    B = set(re.findall(r"[A-Za-z0-9']+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _embed_pair_openai(text_a: str, text_b: str) -> tuple[list[float], list[float]] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    model = os.environ.get("AGVL_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=key)
    r = client.embeddings.create(model=model, input=[text_a, text_b])
    vecs = [list(d.embedding) for d in r.data]
    if len(vecs) != 2:
        return None
    return vecs[0], vecs[1]


def _embed_pair_local(text_a: str, text_b: str) -> tuple[list[float], list[float]] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    name = os.environ.get("AGVL_EMBED_MODEL", "all-MiniLM-L6-v2")
    m = SentenceTransformer(name)
    emb = m.encode([text_a, text_b], show_progress_bar=False)
    return [float(x) for x in emb[0]], [float(x) for x in emb[1]]


class ConsensusChecker:
    def __init__(self, *, claude_model: str | None = None, openai_model: str | None = None) -> None:
        self._claude_model = claude_model or os.environ.get("AGVL_CLAUDE_MODEL")
        self._openai_model = openai_model or os.environ.get("AGVL_OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def check(self, query: str, claude_answer: str) -> dict[str, Any]:
        oa = complete_openai(
            f"Answer briefly and directly (2-6 sentences).\n\nQuestion: {query}",
            model=self._openai_model,
            max_tokens=512,
        )
        if oa.strip().startswith("[stub"):
            return {
                "similarity": 1.0,
                "flagged": False,
                "models": ["claude", "openai_skipped"],
                "openai_answer_excerpt": oa[:500],
                "method": "openai_skipped",
            }

        primary = (claude_answer or "").strip()
        if not primary:
            primary = complete_claude(
                f"Answer briefly and directly (2-6 sentences).\n\nQuestion: {query}",
                model=self._claude_model,
                max_tokens=512,
            )

        vecs = _embed_pair_openai(primary, oa) or _embed_pair_local(primary, oa)
        if vecs:
            sim = _cosine(vecs[0], vecs[1])
            method = "embedding_cosine"
        else:
            sim = _jaccard(primary, oa)
            method = "jaccard_fallback"

        flagged = sim < 0.75
        return {
            "similarity": sim,
            "flagged": flagged,
            "models": ["claude", "openai"],
            "openai_answer_excerpt": oa[:500],
            "method": method,
        }

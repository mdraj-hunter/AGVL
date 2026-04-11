"""S1: InputValidator — query quality, domain role, scoped knowledge docs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of as is was are were been be
    has have had do does did will would could should may might must can
    this that these those it its i you we they he she what which who how
    when where why all any each few more most other some such no not only
    same so than too very just about into through during before after above
    below between under again further then once here there both each few
    hello hey hi oh um
    """.split()
)


def _meaningful_tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    out: list[str] = []
    for w in raw:
        if len(w) < 2 or w in _STOPWORDS:
            continue
        out.append(w)
    return out


def _domain_role(query: str) -> str:
    q = (query or "").lower()
    medical = ("diagnos", "patient", "symptom", "disease", "treatment", "drug", "medical", "health", "clinical", "therapy", "diabetes", "cancer", "hospital")
    legal = ("law", "legal", "contract", "court", "liability", "statute", "plaintiff", "defendant", "tort", "crime", "criminal", "civil")
    financial = ("stock", "bond", "invest", "portfolio", "dividend", "tax", "audit", "financial", "bank", "loan", "mortgage", "budget")
    if any(k in q for k in medical):
        return "medical"
    if any(k in q for k in legal):
        return "legal"
    if any(k in q for k in financial):
        return "financial"
    return "general"


def _scoped_docs(query: str, kb_dir: Path, role: str) -> list[str]:
    tokens = set(_meaningful_tokens(query))
    names: list[str] = []
    if not kb_dir.is_dir():
        return names
    role_hints = {
        "medical": ("med", "health", "clin", "bio", "drug"),
        "legal": ("law", "legal", "stat", "court", "contract"),
        "financial": ("fin", "bank", "tax", "invest", "econ"),
        "general": (),
    }
    hints = role_hints.get(role, ())
    for path in sorted(kb_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".md"} or path.name.startswith("."):
            continue
        low = path.name.lower()
        if tokens and any(t in low for t in tokens):
            names.append(path.name)
            continue
        if hints and any(h in low for h in hints):
            names.append(path.name)
    return names[:20]


class InputValidator:
    """Validates a user query, detects vagueness, assigns a domain role, scopes docs."""

    def __init__(self, query: str, *, kb_dir: Path | None = None) -> None:
        self._raw = query if isinstance(query, str) else str(query)
        self._kb_dir = kb_dir or Path(__file__).resolve().parent.parent / "knowledge_base"

    def validate(self) -> dict[str, Any]:
        q = self._raw.strip()
        tokens = _meaningful_tokens(q)
        vague = len(tokens) < 5
        role = _domain_role(q)
        confidence_threshold = 0.45 if vague else 0.82
        scoped_docs = _scoped_docs(q, self._kb_dir, role)
        return {
            "query": q,
            "role": role,
            "confidence_threshold": confidence_threshold,
            "scoped_docs": scoped_docs,
            "vague": vague,
            "meaningful_word_count": len(tokens),
        }

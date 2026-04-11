"""S4: UncertaintyScorer — sentence-level confidence via secondary Claude (or heuristic)."""

from __future__ import annotations

import os
import re
from typing import Any

from components.tool_llm import complete_claude
from stages.json_utils import parse_json_block


_SCORER_SYSTEM = """You estimate per-sentence confidence that each sentence is well-supported and non-hallucinated given no external tools.
Return ONLY JSON: {"sentences":[{"text":"<exact sentence>","confidence":0.0-1.0}], "notes":""}
Use roughly one entry per grammatical sentence. Confidence 1.0 = highly reliable, 0.0 = unreliable.
Flag obvious speculation as low confidence."""


def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p for p in parts if p.strip()]


class UncertaintyScorer:
    def score(self, llm_response: str, *, model: str | None = None) -> dict[str, Any]:
        sentences_in = _split_sentences(llm_response)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if key and sentences_in:
            payload = (
                "Score each sentence in the following answer for epistemic confidence.\n\n"
                f"ANSWER:\n{llm_response}\n"
            )
            raw = complete_claude(payload, system=_SCORER_SYSTEM, model=model, max_tokens=2048)
            parsed = parse_json_block(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("sentences"), list):
                rows: list[dict[str, Any]] = []
                for item in parsed["sentences"]:
                    if not isinstance(item, dict):
                        continue
                    txt = str(item.get("text", "")).strip()
                    try:
                        conf = float(item.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                    conf = max(0.0, min(1.0, conf))
                    rows.append(
                        {
                            "text": txt,
                            "confidence": conf,
                            "flagged": conf < 0.6,
                        }
                    )
                if rows:
                    agg = sum(r["confidence"] for r in rows) / len(rows)
                    return {
                        "sentences": rows,
                        "aggregate_score": agg,
                        "method": "claude_secondary",
                    }

        rows = []
        for s in sentences_in or [llm_response or "(empty)"]:
            conf = 0.72 if len(s) < 120 else 0.55
            rows.append({"text": s, "confidence": conf, "flagged": conf < 0.6})
        agg = sum(r["confidence"] for r in rows) / max(len(rows), 1)
        return {"sentences": rows, "aggregate_score": agg, "method": "heuristic"}

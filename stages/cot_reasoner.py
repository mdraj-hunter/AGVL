"""S3: ChainOfThoughtReasoner — Claude with forced step-by-step reasoning + RAG context."""

from __future__ import annotations

import os
from typing import Any

from components.tool_llm import complete_claude
from stages.json_utils import parse_json_block


_COT_SYSTEM = """You are a careful analytical assistant.
You MUST reason in explicit numbered steps before giving a final answer.
After your steps, output a JSON object ONLY (no markdown fences) with exactly two keys:
"reasoning_trace" (string, your full step-by-step reasoning as plain text) and
"final_answer" (string, concise user-facing answer that respects the retrieved context).
Do not invent facts not supported by the provided context when context is non-empty."""


class ChainOfThoughtReasoner:
    def __init__(self, *, model: str | None = None) -> None:
        self._model = model or os.environ.get("AGVL_CLAUDE_MODEL")

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, ch in enumerate(chunks or [], start=1):
            src = ch.get("source") or (ch.get("metadata") or {}).get("source", "unknown")
            text = ch.get("text", "")
            parts.append(f"[{i}] source={src}\n{text}")
        return "\n\n".join(parts) if parts else "(no retrieved documents)"

    def run(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, str]:
        ctx_block = self._format_context(retrieved_chunks)
        user = (
            f"User query:\n{query}\n\n"
            f"Retrieved context:\n{ctx_block}\n\n"
            "Follow the system instructions and return ONLY the JSON object."
        )
        raw = complete_claude(user, system=_COT_SYSTEM, model=self._model, max_tokens=2048)
        parsed = parse_json_block(raw)
        if isinstance(parsed, dict) and "reasoning_trace" in parsed and "final_answer" in parsed:
            return {
                "reasoning_trace": str(parsed["reasoning_trace"]),
                "final_answer": str(parsed["final_answer"]),
            }
        return {
            "reasoning_trace": raw.strip(),
            "final_answer": raw.strip()[:2000],
        }

"""S5: CriticModel — second Claude pass for fact-check style review."""

from __future__ import annotations

import os
from typing import Any

from components.tool_llm import complete_claude
from stages.json_utils import parse_json_block


_CRITIC_SYSTEM = """You are a fact-checker. Review the assistant response for logical errors, unsupported claims, and citation accuracy relative to any sources implied in the text.
Return ONLY JSON with keys: flagged_claims (array of strings, each a short problematic claim or empty array), verdict (string: pass|warn|fail)."""


class CriticModel:
    def __init__(self, *, model: str | None = None) -> None:
        self._model = model or os.environ.get("AGVL_CLAUDE_MODEL")

    def review(self, assistant_output: str) -> dict[str, Any]:
        user = (
            "Review the following assistant output.\n\n"
            f"ASSISTANT_OUTPUT:\n{assistant_output}\n"
        )
        raw = complete_claude(user, system=_CRITIC_SYSTEM, model=self._model, max_tokens=1024)
        parsed = parse_json_block(raw)
        if isinstance(parsed, dict):
            claims = parsed.get("flagged_claims", [])
            verdict = str(parsed.get("verdict", "warn"))
            if isinstance(claims, list):
                claims = [str(c) for c in claims]
            else:
                claims = [str(claims)]
            return {"flagged_claims": claims, "verdict": verdict, "raw": raw}
        return {"flagged_claims": [], "verdict": "warn", "raw": raw}

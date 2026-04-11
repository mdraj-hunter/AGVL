"""
Anthropic Claude via official SDK. Without ``ANTHROPIC_API_KEY``, returns a stub
string so the pipeline and tests stay offline-friendly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def complete_claude(
    user_message: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return f"[stub: set ANTHROPIC_API_KEY] {user_message[:200]}"
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - optional install
        return f"[stub: pip install anthropic] {e}"

    client = anthropic.Anthropic(api_key=key)
    m = model or os.environ.get("AGVL_CLAUDE_MODEL", "claude-sonnet-4-20250514")
    kwargs: dict[str, Any] = {
        "model": m,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts) or ""


def complete_openai(
    user_message: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return f"[stub: set OPENAI_API_KEY] {user_message[:200]}"
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        return f"[stub: pip install openai] {e}"

    m = model or os.environ.get("AGVL_OPENAI_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=m,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()


@dataclass
class ToolLLM:
    """Thin wrapper for stages that need a single completion entrypoint."""

    model: str | None = None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return complete_claude(prompt, system=system, model=self.model)

import os

import pytest

from components.tool_llm import complete_claude


def test_complete_claude_stub_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = complete_claude("ping")
    assert "stub" in out.lower() or "ANTHROPIC" in out


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set ANTHROPIC_API_KEY for live Claude test",
)
def test_complete_claude_live():
    out = complete_claude("Reply with exactly: OK", max_tokens=16)
    assert "OK" in out

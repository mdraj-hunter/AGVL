"""S8: HallucinationLogger — append pipeline summary to ``logs/hallucination_log.jsonl``."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
_LOG_NAME = "hallucination_log.jsonl"


def _log_dir(context: dict) -> Path:
    if context.get("log_dir"):
        return Path(context["log_dir"])
    env = os.environ.get("AGVL_LOG_DIR")
    if env:
        return Path(env)
    return _DEFAULT_ROOT / "logs"


def infer_stage_caught(ctx: dict[str, Any]) -> str:
    cons = ctx.get("consensus") or {}
    if cons.get("flagged"):
        return "S6"
    cr = ctx.get("critic_result") or {}
    verdict = str(cr.get("verdict", "")).lower()
    claims = cr.get("flagged_claims") or []
    if verdict == "fail":
        return "S5"
    if isinstance(claims, list) and len(claims) > 0:
        return "S5"
    unc = ctx.get("uncertainty_details") or {}
    for row in unc.get("sentences") or []:
        if row.get("flagged"):
            return "S4"
    return "none"


class HallucinationLogger:
    def append_run(self, context: dict[str, Any]) -> Path:
        log_dir = _log_dir(context)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / _LOG_NAME

        cr = context.get("critic_result") or {}
        claims = cr.get("flagged_claims", [])
        if not isinstance(claims, list):
            claims = [str(claims)]

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": context.get("query"),
            "stage_caught": infer_stage_caught(context),
            "flagged_claims": claims,
            "human_verdict": context.get("human_verdict"),
            "model_used": context.get("model_used")
            or os.environ.get("AGVL_CLAUDE_MODEL", "unknown"),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path.resolve()

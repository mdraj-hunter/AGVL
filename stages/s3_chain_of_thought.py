"""S3: ChainOfThoughtReasoner."""

import os

from stages.cot_reasoner import ChainOfThoughtReasoner


def run(context: dict) -> dict:
    q = str(context.get("validated_input") or context.get("query") or "")
    chunks = context.get("retrieved_chunks") or []
    reasoner = ChainOfThoughtReasoner()
    result = reasoner.run(q, chunks)
    out = dict(context)
    out["cot_trace"] = result["reasoning_trace"]
    out["cot_final_answer"] = result["final_answer"]
    out["model_used"] = os.environ.get("AGVL_CLAUDE_MODEL", "claude-sonnet-4-20250514")
    return out

"""
AGVL: runs S1–S8 in order.

Context keys (evolving dict) include: query, s1, validated_input,
retrieved_chunks, rag_meta, cot_trace, cot_final_answer, model_used,
uncertainty_details, uncertainty_score, critic_result, critic_feedback,
consensus, human_domain, human_verdict, human_approved, hallucination_flags,
monitoring_log_path, log_dir (optional override for S8).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

from stages import s1_input_validation as s1
from stages import s2_rag_retrieval as s2
from stages import s3_chain_of_thought as s3
from stages import s4_uncertainty_scoring as s4
from stages import s5_critic_model as s5
from stages import s6_cross_model_consensus as s6
from stages import s7_human_gate as s7
from stages import s8_monitoring as s8

load_dotenv(_ROOT / ".env")

STAGES: list[tuple[str, object]] = [
    ("s1_input_validation", s1.run),
    ("s2_rag_retrieval", s2.run),
    ("s3_chain_of_thought", s3.run),
    ("s4_uncertainty_scoring", s4.run),
    ("s5_critic_model", s5.run),
    ("s6_cross_model_consensus", s6.run),
    ("s7_human_gate", s7.run),
    ("s8_monitoring", s8.run),
]


def _run_pipeline_core(initial_context: dict | None = None) -> dict:
    ctx: dict = dict(initial_context or {})
    if "query" not in ctx:
        ctx["query"] = os.environ.get("AGVL_DEFAULT_QUERY", "")
    if "hallucination_flags" not in ctx:
        ctx["hallucination_flags"] = []
    for _name, fn in STAGES:
        ctx = fn(ctx)
    return ctx


run_pipeline = _run_pipeline_core
if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    try:
        from langsmith import traceable

        run_pipeline = traceable(name="AGVL_pipeline", run_type="chain")(
            _run_pipeline_core
        )
    except ImportError:
        pass


def main() -> None:
    ctx = run_pipeline({"query": "smoke test"})
    keys = sorted(ctx.keys())
    print("AGVL pipeline finished.")
    print("context keys:", ", ".join(keys))


if __name__ == "__main__":
    main()

"""S4: UncertaintyScorer."""

import os

from stages.uncertainty_scorer import UncertaintyScorer


def run(context: dict) -> dict:
    text = str(context.get("cot_final_answer") or context.get("cot_trace") or "")
    scorer = UncertaintyScorer()
    details = scorer.score(text, model=os.environ.get("AGVL_CLAUDE_MODEL"))
    out = dict(context)
    out["uncertainty_details"] = details
    out["uncertainty_score"] = float(details.get("aggregate_score", 0.5))
    return out

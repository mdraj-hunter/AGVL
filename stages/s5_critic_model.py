"""S5: CriticModel."""

import json

from stages.critic_model import CriticModel


def run(context: dict) -> dict:
    trace = str(context.get("cot_trace", ""))
    ans = str(context.get("cot_final_answer", ""))
    bundle = f"REASONING_TRACE:\n{trace}\n\nFINAL_ANSWER:\n{ans}\n"
    critic = CriticModel()
    result = critic.review(bundle)
    out = dict(context)
    out["critic_result"] = result
    out["critic_feedback"] = json.dumps(
        {"flagged_claims": result.get("flagged_claims"), "verdict": result.get("verdict")},
        ensure_ascii=False,
    )
    return out

"""S6: ConsensusChecker."""

from stages.consensus_checker import ConsensusChecker


def run(context: dict) -> dict:
    q = str(context.get("validated_input") or context.get("query") or "")
    claude_answer = str(context.get("cot_final_answer") or "")
    checker = ConsensusChecker()
    consensus = checker.check(q, claude_answer)
    out = dict(context)
    out["consensus"] = consensus
    return out

"""S7: HumanGate."""

from stages.human_gate_runner import HumanGate


def run(context: dict) -> dict:
    gate = HumanGate()
    hg = gate.run(context)
    out = dict(context)
    out.update(hg)
    out["human_approved"] = hg["human_approved"]
    if hg.get("human_edited_response"):
        out["cot_final_answer"] = hg["human_edited_response"]
    return out

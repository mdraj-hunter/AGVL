"""S8: HallucinationLogger (+ lightweight hallucination_flags for downstream)."""

from pathlib import Path

from stages.hallucination_logger import HallucinationLogger


def run(context: dict) -> dict:
    flags: list[str] = list(context.get("hallucination_flags") or [])
    if float(context.get("uncertainty_score", 0)) < 0.6:
        flags.append("low_aggregate_confidence")
    unc = context.get("uncertainty_details") or {}
    if any((s or {}).get("flagged") for s in unc.get("sentences", [])):
        flags.append("flagged_sentence")
    if (context.get("consensus") or {}).get("flagged"):
        flags.append("consensus_mismatch")
    cr = context.get("critic_result") or {}
    if str(cr.get("verdict", "")).lower() == "fail":
        flags.append("critic_fail")

    path = HallucinationLogger().append_run({**context, "hallucination_flags": flags})

    out = dict(context)
    out["monitoring_log_path"] = str(path)
    out["hallucination_flags"] = flags
    return out

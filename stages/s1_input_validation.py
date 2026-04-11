"""S1: InputValidator."""

from pathlib import Path

from stages.input_validator import InputValidator

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _PACKAGE_ROOT / "knowledge_base"


def run(context: dict) -> dict:
    raw = context.get("query") or ""
    v = InputValidator(raw, kb_dir=_KB_DIR)
    full = v.validate()
    out = dict(context)
    out["s1"] = {
        "query": full["query"],
        "role": full["role"],
        "confidence_threshold": full["confidence_threshold"],
        "scoped_docs": full["scoped_docs"],
    }
    out["s1_meta"] = {
        "vague": full["vague"],
        "meaningful_word_count": full["meaningful_word_count"],
    }
    out["validated_input"] = full["query"]
    return out

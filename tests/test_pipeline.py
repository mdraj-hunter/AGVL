import json
from pathlib import Path

import pytest

from pipeline.agvl_pipeline import STAGES, run_pipeline


def test_run_pipeline_completes_with_expected_keys(monkeypatch):
    monkeypatch.setenv("AGVL_NON_INTERACTIVE", "true")
    q = (
        "Explain differences between civil liability and criminal liability "
        "under common law jurisdictions for a law student"
    )
    ctx = run_pipeline({"query": f"  {q}  "})
    for key in (
        "query",
        "s1",
        "validated_input",
        "retrieved_chunks",
        "rag_meta",
        "cot_trace",
        "cot_final_answer",
        "uncertainty_score",
        "uncertainty_details",
        "critic_feedback",
        "critic_result",
        "consensus",
        "human_approved",
        "human_verdict",
        "hallucination_flags",
        "monitoring_log_path",
        "model_used",
    ):
        assert key in ctx
    assert ctx["validated_input"].strip() == q.strip()
    assert isinstance(ctx["retrieved_chunks"], list)
    assert isinstance(ctx["consensus"], dict)
    assert isinstance(ctx["s1"], dict)
    assert set(ctx["s1"].keys()) >= {"query", "role", "confidence_threshold", "scoped_docs"}


def test_s8_writes_hallucination_log_jsonl(tmp_path, monkeypatch):
    monkeypatch.delenv("AGVL_LOG_DIR", raising=False)
    monkeypatch.setenv("AGVL_NON_INTERACTIVE", "true")
    log_dir = tmp_path / "logs"
    ctx = run_pipeline({"query": "log test with enough meaningful vocabulary words", "log_dir": str(log_dir)})
    log_file = Path(ctx["monitoring_log_path"])
    assert log_file.name == "hallucination_log.jsonl"
    assert log_file.parent == log_dir.resolve()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["query"] == "log test with enough meaningful vocabulary words"
    assert "timestamp" in row
    for k in ("stage_caught", "flagged_claims", "human_verdict", "model_used"):
        assert k in row


def test_eight_stages_registered():
    assert len(STAGES) == 8


def test_input_validator_vague_and_scoped(tmp_path):
    from stages.input_validator import InputValidator

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "legal_notes.md").write_text("civil liability tort", encoding="utf-8")
    v = InputValidator("too short", kb_dir=kb)
    r = v.validate()
    assert r["vague"] is True
    assert r["meaningful_word_count"] < 5

    v2 = InputValidator(
        "Compare civil liability versus criminal liability in contract disputes",
        kb_dir=kb,
    )
    r2 = v2.validate()
    assert r2["vague"] is False
    assert r2["role"] == "legal"
    assert "legal_notes.md" in r2["scoped_docs"]

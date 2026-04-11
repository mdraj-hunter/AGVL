"""S7: HumanGate — domain-aware CLI approval for high-risk queries."""

from __future__ import annotations

import os
import sys
from typing import Any


def _domain_from_query(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ("diagnos", "patient", "treatment", "medical", "drug", "symptom")):
        return "medical"
    if any(k in q for k in ("law", "legal", "contract", "court", "liability", "crime")):
        return "legal"
    if any(k in q for k in ("invest", "stock", "tax", "financial", "bank", "loan")):
        return "financial"
    return "general"


def _effective_domain(query: str, s1_role: str | None) -> str:
    if s1_role in ("medical", "legal", "financial", "general"):
        return s1_role
    return _domain_from_query(query)


def _high_risk(domain: str) -> bool:
    return domain in ("medical", "legal", "financial")


class HumanGate:
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        query = str(context.get("query", ""))
        s1 = context.get("s1") or {}
        role = s1.get("role") if isinstance(s1, dict) else None
        domain = _effective_domain(query, str(role) if role else None)
        response_text = str(
            context.get("cot_final_answer")
            or context.get("validated_input")
            or query
        )

        non_interactive = os.environ.get("AGVL_NON_INTERACTIVE", "").lower() in (
            "1",
            "true",
            "yes",
        ) or not sys.stdin.isatty()

        verdict = "approve"
        edited: str | None = None

        if _high_risk(domain) and not non_interactive:
            print("\n--- AGVL Human Gate (high-risk domain) ---")
            print(f"Domain: {domain}")
            print("Proposed model response:\n")
            print(response_text[:8000])
            print("\nType: approve | reject | edit")
            line = (input("> ").strip().lower() or "reject")
            if line.startswith("approve"):
                verdict = "approve"
            elif line.startswith("edit"):
                verdict = "edit"
                edited = (input("Edited response: ").strip() or "").strip()
            else:
                verdict = "reject"
        elif _high_risk(domain) and non_interactive:
            verdict = (
                os.environ.get("AGVL_HUMAN_VERDICT", "approve").strip().lower() or "approve"
            )
            if verdict.startswith("reject"):
                verdict = "reject"
            elif verdict.startswith("edit"):
                verdict = "edit"
                edited = os.environ.get("AGVL_HUMAN_EDITED_RESPONSE", "").strip() or None
            else:
                verdict = "approve"

        if verdict == "approve":
            approved = True
        elif verdict == "edit":
            approved = bool(edited)
        else:
            approved = False

        return {
            "human_domain": domain,
            "human_verdict": verdict,
            "human_approved": approved,
            "human_edited_response": edited,
        }

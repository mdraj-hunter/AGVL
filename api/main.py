"""
FastAPI entrypoint: POST JSON body with ``query`` to run the AGVL pipeline.

Run from ``agvl-pipeline/``::

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv(_ROOT / ".env")

from pipeline.agvl_pipeline import run_pipeline  # noqa: E402

app = FastAPI(title="AGVL", version="0.1.0")


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to run through S1–S8")


class RunResponse(BaseModel):
    context: dict


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/run", response_model=RunResponse)
def run_agvl(body: RunRequest) -> RunResponse:
    ctx = run_pipeline({"query": body.query})
    return RunResponse(context=ctx)

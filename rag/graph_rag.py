"""
Graph-RAG (S2 adjunct): LangChain ``Neo4jGraph`` + optional Cypher read.
Requires ``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_PASSWORD`` and optional
``AGVL_GRAPH_CYPHER`` (defaults to a small neighbourhood sample).
"""

from __future__ import annotations

import os
from typing import Any


def graph_chunks_for_query(query: str) -> list[dict[str, Any]]:
    if os.environ.get("AGVL_GRAPH_RAG", "").lower() not in ("1", "true", "yes"):
        return []

    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "neo4j").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not uri or not password:
        return []

    try:
        from langchain_community.graphs import Neo4jGraph
    except ImportError:
        return []

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password)
    except Exception:  # pragma: no cover - network / auth
        return []

    cypher = os.environ.get(
        "AGVL_GRAPH_CYPHER",
        "MATCH (n) WITH n LIMIT 25 RETURN labels(n) AS labels, properties(n) AS props",
    )
    try:
        data = graph.query(cypher)
    except Exception:  # pragma: no cover
        return []

    text = f"graph_context query={query!r} rows={data!r}"
    return [{"source": "neo4j", "text": text[:12000], "graph": True}]

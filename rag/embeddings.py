"""Optional sentence-transformers embeddings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_MODEL = None


def get_sentence_transformer() -> Any | None:
    global _MODEL
    if _MODEL is not False:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            _MODEL = False
            return None
        name = os.environ.get("AGVL_EMBED_MODEL", "all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(name)
        return _MODEL
    return None


def embed_texts(texts: list[str]) -> "np.ndarray | None":
    model = get_sentence_transformer()
    if model is None or not texts:
        return None
    import numpy as np

    vectors = model.encode(texts, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")

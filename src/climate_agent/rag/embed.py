from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = Path(__file__).parent / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalized vectors (cosine similarity == dot product).

    Args: texts — strings to embed.
    Returns: (len(texts), embedding_dim) float32 array.
    """
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def save_embeddings(embeddings: np.ndarray) -> None:
    """Persist embeddings to EMBEDDINGS_FILE."""
    np.save(EMBEDDINGS_FILE, embeddings)


def load_embeddings(texts: list[str]) -> np.ndarray:
    """Load cached embeddings if present and the right shape, else compute and cache them.

    Args: texts — the corpus texts these embeddings must correspond to (used only for length
    validation, e.g. after a corpus edit).
    Returns: (len(texts), embedding_dim) float32 array.
    """
    if EMBEDDINGS_FILE.exists():
        cached = np.load(EMBEDDINGS_FILE)
        if cached.shape[0] == len(texts):
            return cached
    embeddings = embed_texts(texts)
    save_embeddings(embeddings)
    return embeddings

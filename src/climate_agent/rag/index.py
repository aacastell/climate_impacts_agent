from functools import lru_cache

import numpy as np

from climate_agent.rag.corpus import load_corpus
from climate_agent.rag.embed import embed_texts, load_embeddings


@lru_cache(maxsize=1)
def _corpus_and_embeddings() -> tuple[list[dict], np.ndarray]:
    """Load the corpus and its embeddings once per process."""
    chunks = load_corpus()
    embeddings = load_embeddings([c["text"] for c in chunks])
    return chunks, embeddings


def search(query: str, top_k: int = 3, tag: str | None = None) -> list[dict]:
    """Retrieve the top_k corpus chunks most similar to a query, via cosine similarity.

    Args: query — natural-language search text. top_k — number of results. tag — if given,
    restrict to chunks whose "tags" list contains this value (e.g. "Agriculture").
    Returns: chunk dicts (text, source_title, source_url, tags), most similar first.
    """
    chunks, embeddings = _corpus_and_embeddings()

    if tag is not None:
        candidates = [(i, c) for i, c in enumerate(chunks) if tag in c["tags"]]
    else:
        candidates = list(enumerate(chunks))
    if not candidates:
        return []

    indices = [i for i, _ in candidates]
    query_vec = embed_texts([query])[0]
    similarities = embeddings[indices] @ query_vec

    ranked = sorted(zip(indices, similarities), key=lambda pair: pair[1], reverse=True)
    return [chunks[i] for i, _ in ranked[:top_k]]

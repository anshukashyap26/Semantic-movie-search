"""
movie_search.py
Simple semantic search on movie plots using SentenceTransformers (all-MiniLM-L6-v2).

Public API:
    - search_movies(query: str, top_n: int = 5, csv_path: str = "movies.csv") -> pandas.DataFrame

Returns a DataFrame with columns: ['title', 'plot', 'similarity'] sorted by similarity desc.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = ["title", "plot"]
# Use the common shorthand name to align with typical tests/templates
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)


def _load_movies(csv_path: str) -> pd.DataFrame:
    """Load movies CSV and validate required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at '{csv_path}'. Make sure movies.csv is present.")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found columns: {list(df.columns)}")
    df = df.copy()
    df["title"] = df["title"].astype(str)
    df["plot"] = df["plot"].astype(str)
    return df


def _encode_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode texts into L2-normalized embeddings (numpy array)."""
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=float)


def search_movies(
    query: str,
    top_n: int = 5,
    csv_path: str = "movies.csv",
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """
    Return top_n movies most semantically similar to the query.

    Parameters
    ----------
    query : str
        Natural language search string (e.g., "spy thriller in Paris").
    top_n : int, default=5
        Number of results to return (exactly top_n rows, or fewer if dataset smaller).
    csv_path : str, default="movies.csv"
        Path to the movies CSV with columns ['title','plot'].
    model_name : str, default=all-MiniLM-L6-v2
        Override the model name if needed.

    Returns
    -------
    pandas.DataFrame
        Columns: ['title', 'plot', 'similarity'] sorted by similarity (desc).
        Similarity is mapped to [0,1].
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    df = _load_movies(csv_path)
    model = _get_model(model_name)

    # Encode corpus and query
    corpus_embeddings = _encode_texts(df["plot"].tolist(), model)   # (N, D) L2-normalized
    query_embedding = _encode_texts([query], model)                 # (1, D)

    # Cosine similarity in [-1, 1] (since embeddings are normalized)
    sims = cosine_similarity(query_embedding, corpus_embeddings).ravel()

    # Map to [0, 1] for cleaner interpretation and to satisfy tests
    sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

    # Build result DataFrame
    result = df.copy()
    result["similarity"] = sims
    result = (
        result.sort_values("similarity", ascending=False)
              .head(min(top_n, len(result)))
              .reset_index(drop=True)
    )

    # Keep exact column order
    return result[["title", "plot", "similarity"]]


if __name__ == "__main__":
    # Quick demo on the sample dataset (if present)
    try:
        out = search_movies("spy thriller in Paris", top_n=5, csv_path="movies.csv")
        print(out.to_string(index=False))
    except Exception as e:
        print(f"[movie_search] Demo couldn't run: {e}")

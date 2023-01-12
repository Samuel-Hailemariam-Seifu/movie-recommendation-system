"""Core recommendation logic and artifact-backed recommender class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MOVIES_PROCESSED_PATH, SIMILARITY_PATH, VECTORIZER_PATH


def build_similarity_model(
    movies_df: pd.DataFrame, max_features: int = 10000
) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fit a TF-IDF model and compute cosine similarity matrix."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1, 2))
    tag_matrix = vectorizer.fit_transform(movies_df["tags"].fillna(""))
    similarity = cosine_similarity(tag_matrix)
    return vectorizer, similarity


def get_popular_fallback(movies_df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Popularity-based recommendations used for weak or missing matches."""
    ranked = movies_df.sort_values(
        by=["vote_average", "vote_count", "popularity"], ascending=False
    ).head(top_n)
    return [
        {
            "title": row["title"],
            "similarity_score": None,
            "genres": row.get("genres_display", "Unknown"),
            "overview_snippet": row.get("overview_snippet", ""),
            "reason": "Popular fallback based on ratings and popularity.",
        }
        for _, row in ranked.iterrows()
    ]


def recommend_movies(
    title: str,
    movies_df: pd.DataFrame,
    similarity: np.ndarray,
    top_n: int = 5,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Recommend similar movies based on cosine similarity scores.

    If the title does not exist or similarity is too weak, return popular fallback movies.
    """
    if not title:
        return get_popular_fallback(movies_df, top_n=top_n)

    normalized_title = title.strip().lower()
    title_to_index = pd.Series(
        movies_df.index, index=movies_df["title"].astype(str).str.strip().str.lower()
    ).drop_duplicates()
    movie_index = title_to_index.get(normalized_title)

    if movie_index is None:
        return get_popular_fallback(movies_df, top_n=top_n)

    distances = similarity[movie_index]
    ranked_indices = np.argsort(distances)[::-1]
    ranked_indices = [idx for idx in ranked_indices if idx != movie_index]

    recs: list[dict[str, Any]] = []
    for idx in ranked_indices:
        score = float(distances[idx])
        # Keep only meaningful positive similarity in model-driven results.
        if score <= min_similarity:
            continue
        row = movies_df.iloc[idx]
        recs.append(
            {
                "title": row["title"],
                "similarity_score": round(score, 4),
                "genres": row.get("genres_display", "Unknown"),
                "overview_snippet": row.get("overview_snippet", ""),
                "reason": (
                    "Recommended because it shares genre, themes, cast, or storyline patterns."
                ),
            }
        )
        if len(recs) >= top_n:
            break

    if not recs:
        return get_popular_fallback(movies_df, top_n=top_n)

    if len(recs) < top_n:
        fallback = get_popular_fallback(movies_df, top_n=top_n * 2)
        existing_titles = {item["title"] for item in recs}
        for item in fallback:
            if item["title"] not in existing_titles and item["title"].lower() != normalized_title:
                recs.append(item)
            if len(recs) >= top_n:
                break

    return recs


@dataclass
class MovieRecommender:
    """Artifact-backed recommender service."""

    movies_df: pd.DataFrame
    similarity: np.ndarray

    @classmethod
    def load(
        cls,
        movies_path: str = str(MOVIES_PROCESSED_PATH),
        similarity_path: str = str(SIMILARITY_PATH),
    ) -> "MovieRecommender":
        movies_df = pd.read_pickle(movies_path)
        similarity = joblib.load(similarity_path)
        return cls(movies_df=movies_df, similarity=similarity)

    def recommend(self, title: str, top_n: int = 5) -> list[dict[str, Any]]:
        return recommend_movies(title=title, movies_df=self.movies_df, similarity=self.similarity, top_n=top_n)

    @property
    def movie_titles(self) -> list[str]:
        return sorted(self.movies_df["title"].dropna().unique().tolist())


def save_artifacts(
    vectorizer: TfidfVectorizer,
    similarity: np.ndarray,
    movies_df: pd.DataFrame,
) -> None:
    """Persist model artifacts for app and prediction use."""
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(similarity, SIMILARITY_PATH)
    movies_df.to_pickle(MOVIES_PROCESSED_PATH)


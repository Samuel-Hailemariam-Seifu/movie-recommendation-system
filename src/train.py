"""Training entrypoint for preprocessing, modeling, and report generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DATASET_SUMMARY_PATH,
    GENRE_DISTRIBUTION_PNG_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_RECOMMENDATIONS_PATH,
    TOP_MOVIES_PNG_PATH,
)
from src.data_loader import load_tmdb_data
from src.preprocess import preprocess_movies
from src.recommender import build_similarity_model, recommend_movies, save_artifacts
from src.utils import ensure_dir, save_json


def _plot_top_movies(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    top_movies = (
        df.sort_values(by=["vote_average", "vote_count"], ascending=False)
        .head(top_n)[["title", "vote_average"]]
        .sort_values(by="vote_average")
    )
    plt.figure(figsize=(10, 6))
    plt.barh(top_movies["title"], top_movies["vote_average"])
    plt.title("Top Movies by Average Rating")
    plt.xlabel("Vote Average")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_genres(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> list[tuple[str, int]]:
    genre_tokens = []
    for genres in df["genres_display"].fillna(""):
        genre_tokens.extend([g.strip() for g in genres.split(",") if g.strip() and g.strip() != "Unknown"])
    genre_counts = Counter(genre_tokens)
    most_common = genre_counts.most_common(top_n)

    plt.figure(figsize=(10, 6))
    labels = [name for name, _ in most_common]
    values = [count for _, count in most_common]
    plt.bar(labels, values)
    plt.title("Top Genres in Dataset")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return most_common


def _get_top_tokens(df: pd.DataFrame, top_n: int = 20) -> list[tuple[str, int]]:
    token_counter = Counter(" ".join(df["tags"].fillna("")).split())
    return token_counter.most_common(top_n)


def main() -> None:
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    raw_df = load_tmdb_data()
    processed_df = preprocess_movies(raw_df)

    vectorizer, similarity = build_similarity_model(processed_df)
    save_artifacts(vectorizer, similarity, processed_df)

    processed_df.to_csv(PROCESSED_DATA_DIR / "movies_processed.csv", index=False)

    top_genres = _plot_genres(processed_df, GENRE_DISTRIBUTION_PNG_PATH)
    _plot_top_movies(processed_df, TOP_MOVIES_PNG_PATH)
    top_tokens = _get_top_tokens(processed_df)

    samples = ["Avatar", "The Dark Knight", "Inception"]
    sample_recs = {}
    for title in samples:
        sample_recs[title] = recommend_movies(
            title=title, movies_df=processed_df, similarity=similarity, top_n=5
        )
    save_json(sample_recs, SAMPLE_RECOMMENDATIONS_PATH)

    summary = {
        "num_movies_processed": int(processed_df.shape[0]),
        "num_features": int(processed_df.shape[1]),
        "vectorizer_type": "TfidfVectorizer",
        "similarity_metric": "cosine_similarity",
        "top_genres": [{"genre": g, "count": c} for g, c in top_genres],
        "top_tokens": [{"token": t, "count": c} for t, c in top_tokens],
    }
    save_json(summary, DATASET_SUMMARY_PATH)

    print(json.dumps({"status": "ok", "movies_processed": int(processed_df.shape[0])}, indent=2))


if __name__ == "__main__":
    main()


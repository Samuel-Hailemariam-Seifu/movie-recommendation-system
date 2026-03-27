"""Preprocessing logic for content-based movie recommendations."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from src.utils import clean_text


def _parse_json_list(raw_value: Any) -> list[dict[str, Any]]:
    """Safely parse TMDB JSON-like list fields from CSV strings."""
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, list):
        return raw_value
    try:
        parsed = ast.literal_eval(raw_value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _extract_names(raw_value: Any, top_k: int | None = None) -> list[str]:
    items = _parse_json_list(raw_value)
    names = [item.get("name", "").strip() for item in items if isinstance(item, dict)]
    names = [name for name in names if name]
    if top_k is not None:
        names = names[:top_k]
    return names


def _extract_director(raw_value: Any) -> str:
    items = _parse_json_list(raw_value)
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name", "").strip()
    return ""


def preprocess_movies(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform TMDB data into model-ready features.

    Returns:
        Processed dataframe with rich text tags for vectorization.
    """
    required_columns = [
        "title",
        "genres",
        "keywords",
        "overview",
        "cast",
        "crew",
        "vote_average",
        "vote_count",
        "popularity",
        "release_date",
    ]
    missing = [col for col in required_columns if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw_df[required_columns].copy()
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df["overview"] = df["overview"].fillna("")

    df["genres_list"] = df["genres"].apply(_extract_names)
    df["keywords_list"] = df["keywords"].apply(_extract_names)
    df["cast_list"] = df["cast"].apply(lambda value: _extract_names(value, top_k=3))
    df["director"] = df["crew"].apply(_extract_director)

    def make_tags(row: pd.Series) -> str:
        tokens: list[str] = []
        tokens.extend(row["genres_list"])
        tokens.extend(row["keywords_list"])
        tokens.extend(row["cast_list"])
        if row["director"]:
            tokens.append(row["director"])
        tokens.extend(str(row["overview"]).split())
        return clean_text(" ".join(tokens))

    df["tags"] = df.apply(make_tags, axis=1)
    df["genres_display"] = df["genres_list"].apply(lambda g: ", ".join(g) if g else "Unknown")
    df["overview_snippet"] = df["overview"].apply(
        lambda text: text[:180].strip() + ("..." if len(text) > 180 else "")
    )

    keep_cols = [
        "title",
        "genres_display",
        "overview",
        "overview_snippet",
        "director",
        "vote_average",
        "vote_count",
        "popularity",
        "release_date",
        "tags",
    ]
    return df[keep_cols].copy()


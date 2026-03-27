"""Load and validate raw movie dataset files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CREDITS_CSV, MOVIES_CSV


def _json_list(items: list[dict]) -> str:
    return json.dumps(items)


def load_demo_data() -> pd.DataFrame:
    """Small built-in dataset so the project runs even without external CSVs."""
    rows = [
        {
            "title": "Interstellar",
            "genres": _json_list([{"name": "Adventure"}, {"name": "Science Fiction"}]),
            "keywords": _json_list([{"name": "space"}, {"name": "time travel"}]),
            "overview": "A team travels through a wormhole in space to ensure humanity survives.",
            "cast": _json_list([{"name": "Matthew McConaughey"}, {"name": "Anne Hathaway"}]),
            "crew": _json_list([{"job": "Director", "name": "Christopher Nolan"}]),
            "vote_average": 8.6,
            "vote_count": 35000,
            "popularity": 120.0,
            "release_date": "2014-11-05",
        },
        {
            "title": "Inception",
            "genres": _json_list([{"name": "Action"}, {"name": "Science Fiction"}]),
            "keywords": _json_list([{"name": "dream"}, {"name": "subconscious"}]),
            "overview": "A skilled thief enters dreams to steal secrets and plant an idea.",
            "cast": _json_list([{"name": "Leonardo DiCaprio"}, {"name": "Joseph Gordon-Levitt"}]),
            "crew": _json_list([{"job": "Director", "name": "Christopher Nolan"}]),
            "vote_average": 8.4,
            "vote_count": 33000,
            "popularity": 110.0,
            "release_date": "2010-07-16",
        },
        {
            "title": "The Dark Knight",
            "genres": _json_list([{"name": "Action"}, {"name": "Crime"}]),
            "keywords": _json_list([{"name": "hero"}, {"name": "vigilante"}]),
            "overview": "Batman faces the Joker, a criminal mastermind spreading chaos in Gotham.",
            "cast": _json_list([{"name": "Christian Bale"}, {"name": "Heath Ledger"}]),
            "crew": _json_list([{"job": "Director", "name": "Christopher Nolan"}]),
            "vote_average": 8.5,
            "vote_count": 32000,
            "popularity": 130.0,
            "release_date": "2008-07-18",
        },
        {
            "title": "The Martian",
            "genres": _json_list([{"name": "Drama"}, {"name": "Science Fiction"}]),
            "keywords": _json_list([{"name": "mars"}, {"name": "astronaut"}]),
            "overview": "An astronaut stranded on Mars must survive until rescue.",
            "cast": _json_list([{"name": "Matt Damon"}, {"name": "Jessica Chastain"}]),
            "crew": _json_list([{"job": "Director", "name": "Ridley Scott"}]),
            "vote_average": 8.0,
            "vote_count": 21000,
            "popularity": 95.0,
            "release_date": "2015-09-30",
        },
        {
            "title": "Titanic",
            "genres": _json_list([{"name": "Drama"}, {"name": "Romance"}]),
            "keywords": _json_list([{"name": "ship"}, {"name": "tragedy"}]),
            "overview": "A romance unfolds aboard the ill-fated Titanic voyage.",
            "cast": _json_list([{"name": "Leonardo DiCaprio"}, {"name": "Kate Winslet"}]),
            "crew": _json_list([{"job": "Director", "name": "James Cameron"}]),
            "vote_average": 7.9,
            "vote_count": 25000,
            "popularity": 85.0,
            "release_date": "1997-12-19",
        },
    ]
    return pd.DataFrame(rows)


def load_tmdb_data(
    movies_path: Path = MOVIES_CSV, credits_path: Path = CREDITS_CSV
) -> pd.DataFrame:
    """
    Load TMDB movies and credits files, then merge into one dataframe.

    Returns:
        DataFrame with movie metadata and credits fields.
    """
    if not movies_path.exists() or not credits_path.exists():
        print(
            "TMDB dataset files not found in data/raw/. "
            "Using built-in demo dataset. Add TMDB CSV files for full results."
        )
        return load_demo_data()

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    merged = movies.merge(credits, on="title", how="inner", suffixes=("", "_credits"))
    merged = merged.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return merged


"""Reusable prediction API for command-line or integration use."""

from __future__ import annotations

import argparse
import json

from src.recommender import MovieRecommender


def recommend_movies_cli(title: str, top_n: int = 5) -> list[dict]:
    recommender = MovieRecommender.load()
    return recommender.recommend(title=title, top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get movie recommendations.")
    parser.add_argument("--title", required=True, help="Movie title to query")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    recommendations = recommend_movies_cli(title=args.title, top_n=args.top_n)
    print(json.dumps(recommendations, indent=2))


if __name__ == "__main__":
    main()


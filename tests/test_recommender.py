import pandas as pd

from src.recommender import build_similarity_model, recommend_movies


def _mock_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Movie A", "Movie B", "Movie C"],
            "genres_display": ["Action", "Action", "Romance"],
            "overview_snippet": ["Hero story", "Hero journey", "Love story"],
            "vote_average": [8.0, 7.8, 7.9],
            "vote_count": [1000, 800, 1200],
            "popularity": [100, 90, 95],
            "tags": ["action hero city", "action hero adventure", "romance love relationship"],
        }
    )


def test_recommend_output_shape() -> None:
    movies = _mock_movies_df()
    _, similarity = build_similarity_model(movies)
    recs = recommend_movies("Movie A", movies, similarity, top_n=2)

    assert isinstance(recs, list)
    assert len(recs) == 2
    assert "title" in recs[0]


def test_missing_title_fallback() -> None:
    movies = _mock_movies_df()
    _, similarity = build_similarity_model(movies)
    recs = recommend_movies("Unknown Movie", movies, similarity, top_n=2)

    assert len(recs) == 2
    assert recs[0]["similarity_score"] is None


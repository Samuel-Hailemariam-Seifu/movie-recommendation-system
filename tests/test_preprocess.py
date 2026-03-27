import pandas as pd

from src.preprocess import preprocess_movies


def test_preprocess_outputs_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "title": ["Movie A"],
            "genres": ['[{"id": 1, "name": "Action"}]'],
            "keywords": ['[{"id": 10, "name": "hero"}]'],
            "overview": ["An action hero saves the city."],
            "cast": ['[{"name": "Actor One"}, {"name": "Actor Two"}]'],
            "crew": ['[{"job": "Director", "name": "Director One"}]'],
            "vote_average": [7.5],
            "vote_count": [1000],
            "popularity": [50.0],
            "release_date": ["2010-01-01"],
        }
    )

    processed = preprocess_movies(df)

    assert "tags" in processed.columns
    assert "genres_display" in processed.columns
    assert processed.shape[0] == 1
    assert "action" in processed.iloc[0]["tags"]


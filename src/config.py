"""Application configuration and shared paths."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Preferred TMDB 5000 dataset file names
MOVIES_CSV = RAW_DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = RAW_DATA_DIR / "tmdb_5000_credits.csv"

# Artifact paths
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
SIMILARITY_PATH = MODELS_DIR / "similarity.joblib"
MOVIES_PROCESSED_PATH = MODELS_DIR / "movies_processed.pkl"
SAMPLE_RECOMMENDATIONS_PATH = REPORTS_DIR / "sample_recommendations.json"
DATASET_SUMMARY_PATH = REPORTS_DIR / "dataset_summary.json"
TOP_MOVIES_PNG_PATH = REPORTS_DIR / "top_movies.png"
GENRE_DISTRIBUTION_PNG_PATH = REPORTS_DIR / "genre_distribution.png"

# Defaults
DEFAULT_TOP_N = 5
POPULARITY_FALLBACK_N = 10


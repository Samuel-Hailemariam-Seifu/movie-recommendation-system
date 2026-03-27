# Movie Recommendation System

A beginner-friendly, portfolio-ready movie recommendation web app built with Python and Streamlit.

This project recommends similar movies using **content-based filtering** over metadata like genres, keywords, cast, director, and plot overview.

## Problem Statement

Users often struggle to discover movies similar to ones they already like.  
This project solves that by computing feature similarity between movies and returning high-quality recommendations with interpretable reasoning.

## Recommendation Approach

The app uses a content-based pipeline:

1. Load TMDB movie metadata and credits.
2. Parse and clean relevant columns.
3. Combine features into a single `tags` text field:
   - genres
   - keywords
   - top cast
   - director
   - overview terms
4. Convert tags into vectors with `TfidfVectorizer`.
5. Compute movie-to-movie cosine similarity.
6. Return top-N similar titles.
7. If title is missing or similarity is weak, return a popularity-based fallback.

## Dataset

This project uses the **TMDB 5000 Movie Dataset**:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place both files inside:

`data/raw/`

Expected paths:

- `data/raw/tmdb_5000_movies.csv`
- `data/raw/tmdb_5000_credits.csv`

## Project Structure

```text
movie-recommendation-system/
  data/
    raw/
    processed/
  notebooks/
    eda.ipynb
  src/
    __init__.py
    config.py
    data_loader.py
    preprocess.py
    recommender.py
    train.py
    predict.py
    utils.py
  app/
    streamlit_app.py
  models/
    vectorizer.joblib
    similarity.joblib
    movies_processed.pkl
  reports/
    dataset_summary.json
    top_movies.png
    genre_distribution.png
  tests/
    test_preprocess.py
    test_recommender.py
  requirements.txt
  README.md
  .gitignore
```

## Setup and Run

### 1) Create environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Add dataset files

Download TMDB 5000 CSVs and place them in `data/raw/`.

If CSV files are missing, the training script automatically uses a small built-in demo dataset so the app still runs end-to-end.

### 3) Train artifacts and generate reports

```bash
python -m src.train
```

This creates:

- `models/vectorizer.joblib`
- `models/similarity.joblib`
- `models/movies_processed.pkl`
- `reports/dataset_summary.json`
- `reports/top_movies.png`
- `reports/genre_distribution.png`
- `reports/sample_recommendations.json`

### 4) Launch app

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Features

- Movie select dropdown
- Optional free-text title search
- Slider for number of recommendations
- Recommend button
- Random movie helper button
- Recommendation cards showing:
  - title
  - similarity score
  - genres
  - overview snippet
  - explanation reason
- Sidebar with dataset stats and method summary

## Training Outputs for Demonstration

The training script produces practical recommendation-system outputs:

- dataset summary JSON
- top genres chart
- top rated movies chart
- common tag tokens
- sample recommendation examples

## Testing

Run lightweight tests:

```bash
pytest -q
```

Included tests validate:

- preprocessing output shape/columns
- recommendation output structure
- graceful fallback for unknown titles

## Screenshots

Add screenshots here after running the app:

- `docs/screenshots/home.png` (placeholder)
- `docs/screenshots/recommendations.png` (placeholder)

## Future Improvements

- add posters and metadata from TMDB API
- add hybrid recommendations (content + collaborative filtering)
- add user preference profiles and saved favorites
- add ANN search for scalability
- add deployment (Streamlit Community Cloud, Docker, or cloud VM)


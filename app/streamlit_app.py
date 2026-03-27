"""Streamlit frontend for the movie recommendation system."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import DATASET_SUMMARY_PATH, SAMPLE_RECOMMENDATIONS_PATH
from src.recommender import MovieRecommender


@st.cache_resource
def load_recommender() -> MovieRecommender:
    return MovieRecommender.load()


@st.cache_data
def load_summary(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_sample_recommendations(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def render_recommendation_card(item: dict) -> None:
    score = item.get("similarity_score")
    score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A (fallback)"

    with st.container(border=True):
        st.subheader(item.get("title", "Unknown"))
        st.write(f"**Similarity score:** {score_text}")
        st.write(f"**Genres:** {item.get('genres', 'Unknown')}")
        snippet = item.get("overview_snippet", "No overview available.")
        st.write(f"**Overview:** {snippet}")
        st.caption(item.get("reason", ""))


def main() -> None:
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Recommendation System")
    st.write(
        "Get movie suggestions using content-based filtering on genres, keywords, cast, director, and overview text."
    )

    try:
        recommender = load_recommender()
    except FileNotFoundError:
        st.error("Model artifacts not found. Run `python -m src.train` first.")
        st.stop()

    summary = load_summary(DATASET_SUMMARY_PATH)
    sample_recs = load_sample_recommendations(SAMPLE_RECOMMENDATIONS_PATH)

    with st.sidebar:
        st.header("About")
        st.write("Method: Content-based filtering with TF-IDF + cosine similarity.")
        st.write("Fallback: Popularity/rating-based recommendations when needed.")

        st.header("Dataset Stats")
        if summary:
            st.metric("Movies Processed", summary.get("num_movies_processed", "N/A"))
            top_genres = summary.get("top_genres", [])[:5]
            if top_genres:
                st.write("Top genres:")
                for item in top_genres:
                    st.write(f"- {item['genre']}: {item['count']}")
        else:
            st.info("Run training to generate dataset summary.")

    col1, col2 = st.columns([2, 1])
    with col1:
        movie_title = st.selectbox("Select a movie", recommender.movie_titles)
    with col2:
        top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)

    custom_search = st.text_input("Or type a movie title manually")
    query_title = custom_search.strip() if custom_search.strip() else movie_title

    c1, c2 = st.columns([1, 1])
    with c1:
        run_button = st.button("Recommend", type="primary", use_container_width=True)
    with c2:
        random_button = st.button("Random movie", use_container_width=True)

    if random_button:
        random_title = recommender.movies_df.sample(1)["title"].iloc[0]
        st.info(f"Try this movie: **{random_title}**")

    if run_button:
        recommendations = recommender.recommend(query_title, top_n=top_n)
        st.markdown("### Recommendations")
        st.caption(f"Results for: {query_title}")
        for rec in recommendations:
            render_recommendation_card(rec)

    st.markdown("### How recommendations are generated")
    st.write(
        "Each movie is converted into a single text profile (tags) using genres, keywords, cast, director, "
        "and plot overview. TF-IDF vectorization converts those tags into numeric vectors, and cosine similarity "
        "finds movies with nearby feature patterns."
    )

    if sample_recs:
        st.markdown("### Example outputs from training")
        sample_key = st.selectbox("View precomputed example", list(sample_recs.keys()))
        for rec in sample_recs[sample_key]:
            render_recommendation_card(rec)


if __name__ == "__main__":
    main()


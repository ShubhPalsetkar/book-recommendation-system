from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ModelArtifacts:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: object  # scipy sparse matrix


def load_books(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"id", "title", "author", "genre", "year", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["author"] = df["author"].fillna("").astype(str).str.strip()
    df["genre"] = df["genre"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df


def build_model(df: pd.DataFrame) -> ModelArtifacts:
    # Combine fields to improve recommendations beyond description alone
    combined_text = (
        df["title"] + " " + df["author"] + " " + df["genre"] + " " + df["description"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return ModelArtifacts(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)


def find_best_title_match(df: pd.DataFrame, query: str) -> Optional[int]:
    """
    Returns index of best match (row index) using a simple contains-based match.
    (You can upgrade this to fuzzy matching as a TODO.)
    """
    q = query.strip().lower()
    if not q:
        return None

    # exact match first
    exact = df["title"].str.lower() == q
    if exact.any():
        return int(df[exact].index[0])

    # contains match
    contains = df["title"].str.lower().str.contains(q, na=False)
    if contains.any():
        return int(df[contains].index[0])

    return None


def recommend_by_title(
    artifacts: ModelArtifacts,
    title_query: str,
    top_n: int = 5,
    genre_filter: Optional[str] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    df = artifacts.df
    idx = find_best_title_match(df, title_query)
    if idx is None:
        return pd.DataFrame()

    sims = cosine_similarity(
        artifacts.tfidf_matrix[idx],
        artifacts.tfidf_matrix
    ).flatten()

    rec_df = df.copy()
    rec_df["score"] = sims

    # Exclude the selected book itself
    rec_df = rec_df.drop(index=idx)

    if genre_filter and genre_filter != "All":
        rec_df = rec_df[rec_df["genre"].str.lower() == genre_filter.lower()]

    if year_range:
        y0, y1 = year_range
        rec_df = rec_df[(rec_df["year"] >= y0) & (rec_df["year"] <= y1)]

    rec_df = rec_df.sort_values("score", ascending=False).head(top_n)

    # Keep output clean
    return rec_df[["title", "author", "genre", "year", "score", "description"]]

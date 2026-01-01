import streamlit as st
import pandas as pd

from recommender import load_books, build_model, recommend_by_title

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def load_data():
    # Uses your converted file: data/books.csv
    return load_books("data/books.csv")

@st.cache_resource
def load_model(df: pd.DataFrame):
    return build_model(df)

# -------------------- Load --------------------
df = load_data()
artifacts = load_model(df)

# -------------------- Title --------------------
st.title("📚 Book Recommendation System")
st.write("Find similar books using **TF-IDF vectorization + cosine similarity**.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Filters")

    # Reset button (simple UX win)
    if st.button("Reset filters"):
        st.rerun()

    genres = ["All"] + sorted(df["genre"].dropna().unique().tolist())
    genre_filter = st.selectbox("Genre", genres, index=0)

    # Year range auto-fits dataset (fixes your 2020 threshold issue automatically)
    min_year = int(df["year"].min()) if len(df) else 0
    max_year = int(df["year"].max()) if len(df) else 0
    year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))

    top_n = st.slider("Number of recommendations", 3, 10, 5)

# -------------------- Main content layout --------------------
left, right = st.columns([2, 1], gap="large")

with left:
    # Dataset size
    st.caption(f"📊 Dataset size: **{len(df):,} books**")

    # Dropdown selection (recommended)
    st.subheader("Pick a book (recommended)")

    query = st.text_input(
        "Type to filter titles",
        placeholder="e.g., harry, dune, hobbit"
    ).strip()

    filtered = df[df["title"].str.lower().str.contains(query.lower(), na=False)] if query else df

    # Keep dropdown fast
    options = filtered["title"].head(2000).tolist()
    if not options:
        options = df["title"].head(2000).tolist()

    selected_title = st.selectbox("Select a title", options)

    # Show chosen book info
    book = df[df["title"] == selected_title].head(1)
    if len(book):
        b = book.iloc[0]
        st.info(f"**Selected:** {b['title']} — {b['author']}  |  {b['genre']} • {b['year']}")

    # Recommend button
    if st.button("Recommend"):
        results = recommend_by_title(
            artifacts,
            title_query=selected_title,
            top_n=top_n,
            genre_filter=genre_filter,
            year_range=year_range,
        )

        if results.empty:
            st.warning(
                "No recommendations found with the current filters. "
                "Try widening the **year range**, switching **genre**, or selecting a different book."
            )
        else:
            st.subheader("Recommended books")
            for _, row in results.iterrows():
                st.markdown(f"### {row['title']} — {row['author']}")
                st.caption(f"{row['genre']} • {row['year']} • similarity: {row['score']:.3f}")
                st.write(row["description"])
                st.divider()

with right:
    st.subheader("Insights")

    # Top genres chart
    top_genres = df["genre"].value_counts().head(10)
    st.write("Top genres in the dataset:")
    st.bar_chart(top_genres)

    # Optional: show top-rated books if your converted dataset includes rating
    # (Your Kaggle raw file has average_rating; your books.csv may not—this handles both cases safely.)
    if "average_rating" in df.columns:
        st.write("Top books by average rating:")
        top_rated = (
            df[df["title"].notna()]
            .sort_values("average_rating", ascending=False)
            .head(10)[["title", "author", "genre", "year", "average_rating"]]
        )
        st.dataframe(top_rated, use_container_width=True)
    else:
        st.caption("Tip: If you want a 'Top rated' table, we can include average_rating during dataset conversion.")

st.markdown("---")
st.caption(
    "Built with TF-IDF vectorization and cosine similarity. "
    "Upgrades you can add next: fuzzy matching (typos), embeddings (SBERT), hybrid recommender."
)

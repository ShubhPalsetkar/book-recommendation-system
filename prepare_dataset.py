import os
import glob
import csv
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "books.csv")

def extract_year(date_str):
    """
    Extract year from publication_date.
    Expected formats like: '9/16/2003' or '2003'
    """
    if pd.isna(date_str):
        return 0
    s = str(date_str)
    for token in s.split("/"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return 0

def main():
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        raise SystemExit("No CSV found in data/raw")

    src = files[0]
    df = pd.read_csv(src, low_memory=False)

    print("Loaded file:", src)
    print("Available columns:\n", list(df.columns))

    # Column mapping (based on your dataset)
    col_title = "Title"
    col_author = "Author"
    col_genre = "genres"
    col_pubdate = "publication_date"

    out = pd.DataFrame()
    out["id"] = range(1, len(df) + 1)
    out["title"] = df[col_title].fillna("").astype(str).str.strip()
    out["author"] = df[col_author].fillna("").astype(str).str.strip()

    # Clean genre: keep first genre only
    g = df[col_genre].fillna("").astype(str)
    g = g.str.replace(r"[\[\]\{\}\"']", "", regex=True)
    g = g.str.split(",").str[0].str.strip()
    out["genre"] = g.replace("", "Unknown")

    # Extract year from publication_date
    out["year"] = df[col_pubdate].apply(extract_year)

    # Synthetic description (since dataset has no description)
    out["description"] = (
        "A book titled " + out["title"] +
        " written by " + out["author"] +
        " belonging to the " + out["genre"] + " genre."
    )

    # Remove empty titles
    out = out[out["title"] != ""]

    # Keep app fast
    out = out.head(15000)

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT_PATH, index=False, quoting=csv.QUOTE_ALL)

    print(f"\n✅ Dataset ready: {OUT_PATH}")
    print("Rows:", len(out))

if __name__ == "__main__":
    main()

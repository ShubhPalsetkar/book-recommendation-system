# 📚 Book Recommendation System (TF-IDF + Streamlit)

A content-based book recommender that suggests similar books using **TF-IDF vectorization** and **cosine similarity**.  
Built as a portfolio project to demonstrate an end-to-end workflow: **data preparation → ML pipeline → interactive web app**.

---

## ✨ Features
- ✅ Content-based recommendations (TF-IDF + cosine similarity)
- ✅ Search + dropdown selection (user-friendly)
- ✅ Filters: **Genre** and **Publication year**
- ✅ Top genres visualization
- ✅ Clear UX messages when filters exclude results
- ✅ Clean project structure for portfolio / GitHub

---

## 🧠 How it works (ML)
1. Text features are created from a combination of:
   - title, author, genre, and description (or generated description if missing)
2. A **TF-IDF matrix** is built from the combined text.
3. Similarity between books is calculated using **cosine similarity**.
4. The top-N most similar books are returned and displayed in the Streamlit UI.

---

## 📊 Dataset
This project uses a **Goodreads dataset snapshot** (CSV) containing genres and publication dates.

> **Dataset note:** The provided snapshot includes books up to **~2020** based on available publication dates.  
> Recommendations and year filtering therefore follow the dataset range.

---

## 🗂️ Project Structure
```text
book-recommender/
├── app.py                # Streamlit UI
├── recommender.py        # TF-IDF + cosine similarity logic
├── prepare_dataset.py    # Converts raw dataset into data/books.csv
├── requirements.txt
├── data/
│   ├── books.csv         # Processed dataset used by the app
│   └── raw/              # Raw Kaggle CSV placed here (not required to run if books.csv exists)
└── README.md

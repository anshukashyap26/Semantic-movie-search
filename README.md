Name: Anshu Kashyap
Roll No.: 221010209
Branch: ECE


# Movie Semantic Search Assignment

This repository contains my solution for **Assignment‑1: Semantic Search on Movie Plots** using SentenceTransformers (**all‑MiniLM‑L6‑v2**).

## What this does
- Loads a CSV of movies with `title` and `plot` columns.
- Encodes plots using `all-MiniLM-L6-v2` (SentenceTransformers).
- Implements `search_movies(query, top_n)` to return the top results by cosine similarity.

## Quickstart

```bash
# 1) Clone your repo (example)
git clone https://github.com/anshukashyap26/Semantic-movie-search
cd Semantic-movie-search

# 2) Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Run the included notebook
jupyter notebook movie_search_solution.ipynb
```

## Usage (Python)

```python
from movie_search import search_movies

df = search_movies("spy thriller in Paris", top_n=3, csv_path="movies.csv")
print(df)
```

The function returns a pandas DataFrame with columns: `['title', 'plot', 'score']` sorted by `score` descending.

## Running Unit Tests

The instructor provides tests under `tests/test_movie_search.py`. After copying them into your repo (or if they already exist), run:

```bash
python -m unittest tests/test_movie_search.py -v
```

## Repo Structure (expected)

```
semantic-movie-search/
├─ movie_search.py          # Implementation
├─ movies.csv               # Dataset (ensure it contains 'title' and 'plot')
├─ requirements.txt         # Dependencies
├─ README.md                # This file
├─ movie_search_solution.ipynb  # Walkthrough notebook
└─ tests/
   └─ test_movie_search.py  # Provided by template (copy into your repo)
```

## Notes
- If `movies.csv` is elsewhere, pass its path via `csv_path`.
- The model is cached in memory to avoid reloading on every call.
- Embeddings are L2-normalized; cosine similarity = dot product, ensuring stable ranking.

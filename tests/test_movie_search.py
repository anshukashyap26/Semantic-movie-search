import unittest
import pandas as pd
from movie_search import search_movies

class TestMovieSearch(unittest.TestCase):
    def test_output_format(self):
        df = search_movies("spy thriller in Paris", top_n=3, csv_path="movies.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(all(c in df.columns for c in ["title", "plot", "similarity"]))

    def test_top_n(self):
        top_n = 2
        df = search_movies("spy thriller in Paris", top_n=top_n, csv_path="movies.csv")
        self.assertEqual(len(df), top_n)

    def test_similarity_sorted_and_range(self):
        df = search_movies("spy thriller in Paris", top_n=5, csv_path="movies.csv")
        # sorted high→low
        self.assertTrue(df["similarity"].is_monotonic_decreasing)
        # values clamped to [0,1]
        self.assertTrue(((df["similarity"] >= 0) & (df["similarity"] <= 1)).all())

    def test_non_empty_query(self):
        with self.assertRaises(ValueError):
            search_movies("", top_n=3, csv_path="movies.csv")

if __name__ == "__main__":
    unittest.main()

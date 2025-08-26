import unittest
import pandas as pd
from movie_search import search_movies

class TestMovieSearch(unittest.TestCase):
    def test_output_format(self):
        df = search_movies("spy thriller in Paris", top_n=3, csv_path="movies.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(set(["title", "plot", "score"]).issubset(df.columns))

    def test_top_n(self):
        df = search_movies("spy thriller in Paris", top_n=2, csv_path="movies.csv")
        self.assertLessEqual(len(df), 2)

    def test_similarity_sorted(self):
        df = search_movies("spy thriller in Paris", top_n=5, csv_path="movies.csv")
        self.assertTrue(df["score"].is_monotonic_decreasing)

    def test_non_empty_query(self):
        with self.assertRaises(ValueError):
            search_movies("", top_n=3, csv_path="movies.csv")

if __name__ == "__main__":
    unittest.main()

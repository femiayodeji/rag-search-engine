import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.load_data import MovieData, get_movies, load_data

class TestMoviesLoad(unittest.TestCase):
    def test_data_load(self):
        data = load_data()
        self.assertIsInstance(data, MovieData)
        self.assertIsInstance(data.movies, list)

    def test_get_movies(self):
        movies = get_movies()
        self.assertIsInstance(movies, list)
        if movies:
            self.assertIsInstance(movies[0], dict)
        else:
            self.fail("No movies found in the data.")

if __name__ == "__main__":
    unittest.main()
import unittest
from utils import load_data, get_movies

class TestMoviesLoad(unittest.TestCase):
    def test_data_load(self):
        data = load_data()
        self.assertIsInstance(data, dict)

    def test_get_movies(self):
        movies = get_movies()
        self.assertIsInstance(movies, list)
        if movies:
            self.assertIsInstance(movies[0], dict)
        else:
            self.fail("No movies found in the data.")

if __name__ == "__main__":
    unittest.main()
from dataclasses import dataclass
import json

@dataclass
class MovieData:
    movies: list[dict]

def load_data(path: str = "data/movies.json") -> MovieData:
    with open(path, "r") as file:
           raw = json.load(file)
           return MovieData(movies=raw["movies"])

def get_movies() -> list[dict]:
    data = load_data()
    return data.movies if data.movies else []

def get_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    with open(path, "r") as file:
        return file.read().splitlines()

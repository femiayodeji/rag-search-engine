from collections import Counter
import math
import os
import pickle
from pydoc import doc

from cli.load_data import get_movies
from cli.string_processor import process_str


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[object]] = {}
    def __add_document(self, doc_id, text):
        tokens = process_str(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)
    
    def get_documents(self, term) -> list[int]:
        doc_ids = list(self.index.get(term.lower(), set()))
        sorted_doc_ids = sorted(doc_ids)
        return sorted_doc_ids

    def build(self):
        for movie in get_movies():
            title_description = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], title_description)
            self.docmap[movie["id"]] = movie
    
    def save(self):
        if not os.path.exists("cache"):
            os.mkdir("cache")
        
        pickle.dump(self.index, open("cache/index.pkl", "wb"))
        pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))
        pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))
    
    def load(self):
        try:
            self.index = pickle.load(open("cache/index.pkl", "rb"))
            self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
            self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))
        except FileNotFoundError:
            print("Cache files not found. Please build the index first.")

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str):
        tokens = process_str(term)
        if len(tokens) != 1:
            raise ValueError("The term must be a single token.")
        token = tokens[0]
        document_ids = self.get_documents(token)
        df = len(document_ids)
        N = len(self.docmap)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf
    
def bm25_idf_command(term):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Cache files not found. Please build the index first.")
        return    
    return index.get_bm25_idf(term)
    
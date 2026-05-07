from collections import Counter
import math
import os
import pickle
from pydoc import doc

from cli.load_data import get_movies
from cli.string_processor import process_str

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[object]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join('cache', "doc_lengths.pkl")


    def __add_document(self, doc_id, text):
        tokens = process_str(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)
        self.doc_lengths[doc_id] = len(tokens)
    
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
        pickle.dump(self.doc_lengths, open(self.doc_lengths_path, "wb"))
    
    def load(self):
        try:
            self.index = pickle.load(open("cache/index.pkl", "rb"))
            self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
            self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))
            self.doc_lengths = pickle.load(open(self.doc_lengths_path, "rb"))
        except FileNotFoundError:
            print("Cache files not found. Please build the index first.")

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        doc_count = len(self.doc_lengths)
        return total_length / doc_count if doc_count > 0 else 0.0

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = process_str(term)
        if len(tokens) != 1:
            raise ValueError("The term must be a single token.")
        token = tokens[0]
        document_ids = self.get_documents(token)
        df = len(document_ids)
        N = len(self.docmap)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tokens = process_str(term)
        if len(tokens) != 1:
            raise ValueError("The term must be a single token.")
        token = tokens[0]
        length_norm = (1 - b) + b * (self.doc_lengths.get(doc_id, 0) / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, token)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


def bm25_idf_command(term):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Cache files not found. Please build the index first.")
        return    
    return index.get_bm25_idf(term)
    
def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Cache files not found. Please build the index first.")
        return    
    return index.get_bm25_tf(doc_id, term, k1, b)
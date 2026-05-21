from typing import List
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import Tensor

from cli.load_data import get_movies

class SemanticSearch:
    def __init__(self, model = "All-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Input text cannot be empty.")
        encoded_text = self.model.encode([text])
        return encoded_text[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        docs = []
        for i, doc in enumerate(documents):
            self.document_map[i] = doc
            docs.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(docs, show_progress_bar = True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        docs = []
        for i, doc in enumerate(documents):
            self.document_map[i] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            print("Embeddings loaded from cache.")
            if len(self.embeddings) == len(documents):
                print("Embeddings loaded from cache.")
                return self.embeddings
        else:
            print("Cache file not found. Building embeddings...")
            return self.build_embeddings(documents)

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = get_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")     # type: ignore



def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    if not text:
        raise ValueError("Input text cannot be empty.")
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


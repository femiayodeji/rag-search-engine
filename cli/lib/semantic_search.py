from typing import List
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import Tensor, embedding

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
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append((similarity, self.document_map[i]))
        results.sort(key=lambda x: x[0], reverse=True)

        normalized_results = []
        for score, document in results[:limit]:
            normalized_results.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )
        return normalized_results


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = get_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")     # type: ignore

def embed_query_text(query):
    if not query:
        raise ValueError("Input query cannot be empty.")
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


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


import os
import re
import json

import numpy as np

from cli.lib.semantic_search import SemanticSearch, cosine_similarity

from typing import TypedDict

from cli.search_utils import format_search_result


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        self.chunk_metadata = []
        all_chunks: list[str] = []
        all_metadata: list[ChunkMetadata] = []

        for movie_idx, doc in enumerate(documents):
            self.document_map[movie_idx] = doc
            description = doc["description"]
            if not description:
                continue

            chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            metadata: list[ChunkMetadata] = [
                ChunkMetadata(
                    movie_idx=movie_idx,
                    chunk_idx=chunk_idx,
                    total_chunks=len(chunks),
                )
                for chunk_idx in range(len(chunks))
            ]
            all_metadata.extend(metadata)

        self.chunk_embeddings = np.array(self.model.encode(all_chunks, show_progress_bar=True))
        self.chunk_metadata = all_metadata
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for i, doc in enumerate(documents):
            self.document_map[i] = doc
        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            print("Chunk embeddings loaded from cache.")
            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings
            else:
                print("Cache file found but length mismatch. Rebuilding chunk embeddings...")
                return self.build_chunk_embeddings(documents)
        else:
            print("Cache file not found. Building chunk embeddings...")
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("Chunk embeddings not found. Please build or load chunk embeddings first.")
        
        semantic_search = SemanticSearch()
        query_embedding = semantic_search.generate_embedding(query)

        chunk_scores = []

        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            movie_metadata = self.chunk_metadata[chunk_idx]
            chunk_scores.append({"chunk_idx": chunk_idx, "movie_idx": movie_metadata["movie_idx"], "score": similarity})

        movie_index_to_score = {}
        for score in chunk_scores:
            movie_idx = score["movie_idx"]
            if movie_idx not in movie_index_to_score or score["score"] > movie_index_to_score[movie_idx]:
                movie_index_to_score[movie_idx] = score["score"]
        
        sorted_filtered_movies = sorted(movie_index_to_score.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for movie_idx, score in sorted_filtered_movies:
            document = self.document_map[movie_idx]
            metadata = document
            results.append(format_search_result(movie_idx, document["title"], document["description"], score, metadata))
        return results

        

def semantic_chunk_text(text, max_chunk_size=4, overlap=0):
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_chunk_size")

    sentence_chunks = [
        sentences[i:i + max_chunk_size]
        for i in range(0, len(sentences), step)
        if len(sentences[i:i + max_chunk_size]) > overlap
    ]
    return [" ".join(chunk).strip() for chunk in sentence_chunks if chunk]

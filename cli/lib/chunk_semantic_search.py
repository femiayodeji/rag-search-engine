import os
import re
import json

import numpy as np

from cli.lib.semantic_search import SemanticSearch

from typing import TypedDict


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

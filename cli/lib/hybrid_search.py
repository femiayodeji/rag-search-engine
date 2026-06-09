import os

from cli.inverted_index import InvertedIndex
from cli.lib.chunk_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.doc_lengths_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[dict, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_score(self, rank: int, k: int = 60) -> float:
        return 1 / (k + rank)    

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        if limit <= 0:
            return []

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        normalized_bm25_scores = normalize_scores([score for _, score in bm25_results])
        normalized_semantic_scores = normalize_scores([result["score"] for result in semantic_results])

        document_map = {
            doc["id"]: {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
            for doc in self.documents
        }

        for (document, _), normalized_score in zip(bm25_results, normalized_bm25_scores):
            doc_id = document["id"]
            if doc_id in document_map:
                document_map[doc_id]["bm25_score"] = normalized_score

        for result, normalized_score in zip(semantic_results, normalized_semantic_scores):
            doc_id = result["id"]
            if doc_id in document_map:
                document_map[doc_id]["semantic_score"] = normalized_score

        scored_documents: list[dict] = []
        for document in document_map.values():
            combined_score = hybrid_score(
                document["bm25_score"],
                document["semantic_score"],
                alpha,
            )
            document["hybrid_score"] = combined_score
            document["score"] = combined_score
            scored_documents.append(document)

        scored_documents.sort(key=lambda doc: doc["hybrid_score"], reverse=True)
        return scored_documents[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_ranks = {result["id"]: rank for rank, (result, _) in enumerate(bm25_results, start=1)}
        semantic_ranks = {res["id"]: rank for rank, res in enumerate(semantic_results, start=1)}

        document_map = {
            doc["id"]: {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "bm25_rank": bm25_ranks.get(doc["id"], float("inf")),
                "semantic_rank": semantic_ranks.get(doc["id"], float("inf")),
            }
            for doc in self.documents
        }
        scored_documents: list[dict] = []
        for document in document_map.values():
            rrf_score = self.rrf_score(document["bm25_rank"], k) + self.rrf_score(document["semantic_rank"], k)
            document["score"] = rrf_score
            scored_documents.append(document)
        scored_documents.sort(key=lambda doc: doc["score"], reverse=True)
        return scored_documents[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

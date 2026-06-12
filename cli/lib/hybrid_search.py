from asyncio import sleep
import json
import os

from cli.constants import get_correct_spelling_prompt, get_expand_prompt, get_rerank_prompt_batch, get_rerank_prompt_individual, get_rewrite_prompt
from cli.inverted_index import InvertedIndex
from cli.lib.chunk_semantic_search import ChunkedSemanticSearch
from cli.llm_utils import llm_request
from cli.search_utils import SearchResult, format_search_result

from sentence_transformers import CrossEncoder


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

    def rrf_score(self, rank: int | float, k: int = 60) -> float:
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

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_ranks = {result["id"]: rank for rank, (result, _) in enumerate(bm25_results, start=1)}
        semantic_ranks = {res["id"]: rank for rank, res in enumerate(semantic_results, start=1)}

        scored_documents: list[SearchResult] = []
        for doc in self.documents:
            doc_id = doc["id"]
            bm25_rank = bm25_ranks.get(doc_id, float("inf"))
            semantic_rank = semantic_ranks.get(doc_id, float("inf"))
            rrf_score = self.rrf_score(bm25_rank, k) + self.rrf_score(semantic_rank, k)
            scored_documents.append(
                format_search_result(
                    doc_id=doc_id,
                    title=doc["title"],
                    document=doc["description"],
                    score=rrf_score,
                    metadata={"bm25_rank": bm25_rank, "semantic_rank": semantic_rank},
                )
            )
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

def enhance_query(query: str, method: str) -> str:
    if not method:
        return query
    prompt = get_enhancement_prompt(query, method)
    llm_response = llm_request(prompt)
    enhanced_query = (llm_response.text or "").strip()
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    return enhanced_query

def get_enhancement_prompt(query: str, method: str) -> str:
    match method:
        case "spell":
            return get_correct_spelling_prompt(query)
        case "rewrite":
            return get_rewrite_prompt(query)
        case "expand":
            return get_expand_prompt(query)
        case _:
            raise ValueError(f"Unknown enhancement method: {method}")

async def rerank_llm(query: str, results: list[SearchResult], method: str, limit: int) -> list[SearchResult]:
    if method == "batch":
        doc_list_str = "\n".join([f"{res['title']} - {res['document']}..." for res in results])
        prompt = get_rerank_prompt_batch(query, doc_list_str)
        try:
            llm_response = llm_request(prompt)
            ranked_movie_ids: list[int] = json.loads((llm_response.text or "[]").strip())
            movie_id_to_rank = {movie_id: rank for rank, movie_id in enumerate(ranked_movie_ids, start=1)}
            for result in results:
                result["re_rank_rank"] = movie_id_to_rank.get(result["id"], float("inf"))
            results.sort(key=lambda r: r.get("re_rank_rank", float("inf")))
            return results[:limit]
        except (ValueError, json.JSONDecodeError, RuntimeError):
            for result in results:
                result["re_rank_rank"] = 0.0

    elif method == "individual":
        for result in results:
            prompt = get_rerank_prompt_individual(query, result)
            try:
                llm_response = llm_request(prompt)
                score = float((llm_response.text or "0").strip())
            except (ValueError, RuntimeError):
                score = 0.0
            result["re_rank_score"] = score
            await sleep(3)  # To avoid hitting rate limits
    results.sort(key=lambda r: r.get("re_rank_score", 0.0), reverse=True)
    return results[:limit]

def rerank_cross_encoder(query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
    pairs = [[query, f"{res.get('title', '')} - {res.get('document', '')}"] for res in results]
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    for i, (result, score) in enumerate(zip(results, scores), start=1):
        result["metadata"]["cross_encoder_score"] = score
    results.sort(key=lambda x: x["metadata"].get("cross_encoder_score", 0.0), reverse=True)
    return results[:limit]
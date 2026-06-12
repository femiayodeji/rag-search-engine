#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.load_data import get_movies
from cli.lib.hybrid_search import HybridSearch, rerank_cross_encoder, enhance_query, normalize_scores, rerank_llm

import argparse


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform a weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for BM25 vs semantic score (default: 0.5)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    ranked_parser = subparsers.add_parser("rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) search")
    ranked_parser.add_argument("query", type=str, help="Search query")
    ranked_parser.add_argument("--k", type=int, default=60, help="RRF parameter k (default: 60)")
    ranked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    ranked_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    ranked_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method for reranking results (default: individual)")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return

            normalized = normalize_scores(scores)
            for score in normalized:
                print(f"* {score:.4f}")
                
        case "weighted-search":
            documents = get_movies()
            search = HybridSearch(documents)

            results = search.weighted_search(args.query, alpha=args.alpha, limit=args.limit)

            for i, res in enumerate(results, start=1):
                print(
                    f"{i}. {res['title']} \n"
                    f"Hybrid Score: {res['hybrid_score']:.4f} \n"
                    f"BM25 Score: {res['bm25_score']:.4f} \n"
                    f"Semantic Score: {res['semantic_score']:.4f}\n"
                    f"Description: {res['description']}\n"
                )

        case "rrf-search":
            query = args.query
            
            enhancement_method = args.enhance
            if enhancement_method:
                query = enhance_query(query, enhancement_method)
           
            documents = get_movies()
            search = HybridSearch(documents)

            limit = args.limit
            rerank_method = args.rerank_method

            results = search.rrf_search(query, k=args.k, limit=limit * 5 if rerank_method == "individual" else limit)
            
            if rerank_method:
                print(f"\nRe-ranking top {limit} results using {rerank_method} method...\n")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k=60):\n")
                if rerank_method in ["individual", "batch"]:
                    results = await rerank_llm(query, results, method=rerank_method, limit=limit)
                    for i, res in enumerate(results, start=1):
                        rerank_str = ""
                        if rerank_method == "individual":
                            rerank_str = f"Re-rank Score: {res.get('re_rank_score', 0.0):.3f}/10\n"
                        elif rerank_method == "batch":
                            rerank_str = f"Re-rank Rank: {res.get('re_rank_rank', 0)}\n"
                        print(
                            f"{i}. {res['title']}\n"
                            f"{rerank_str}"
                            f"  RRF Score: {res['score']:.3f}\n"
                            f"  BM25 Rank: {res['metadata']['bm25_rank']}, Semantic Rank: {res['metadata']['semantic_rank']}\n"
                            f"  {res.get('document', '')[:100] + ("..." if len(res.get('document', '')) > 100 else "")}\n"
                        )
                elif rerank_method == "cross_encoder":
                    results = rerank_cross_encoder(query, results, limit=limit)
                    for i, res in enumerate(results, start=1):
                        print(
                            f"{i}. {res['title']}\n"
                            f"  Cross Encoder Score: {res.get('cross_encoder_score', 0.0):.3f}\n"
                            f"  RRF Score: {res['score']:.3f}\n"
                            f"  BM25 Rank: {res['metadata']['bm25_rank']}, Semantic Rank: {res['metadata']['semantic_rank']}\n"
                            f"  {res.get('document', '')[:100] + ("..." if len(res.get('document', '')) > 100 else "")}\n"
                        )
                    
            else:
                for i, res in enumerate(results, start=1):
                    bm25_rank = res["metadata"]["bm25_rank"] if res["metadata"]["bm25_rank"] != float("inf") else "N/A"
                    semantic_rank = res["metadata"]["semantic_rank"] if res["metadata"]["semantic_rank"] != float("inf") else "N/A"
                    print(
                        f"{i}. {res['title']}\n"
                        f"  RRF Score: {res['score']:.3f}\n"
                        f"  BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n"
                        f"  {res.get('document', '')[:100] + ("..." if len(res.get('document', '')) > 100 else "")}\n"
                    )

        case _:
            parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
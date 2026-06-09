#!/usr/bin/env python3
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.load_data import get_movies
from cli.lib.hybrid_search import HybridSearch, normalize_scores

import argparse


def main() -> None:
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
            documents = get_movies()
            search = HybridSearch(documents)

            results = search.rrf_search(args.query, k=args.k, limit=args.limit)

            for i, res in enumerate(results, start=1):
                bm25_rank = res["bm25_rank"] if res["bm25_rank"] != float("inf") else "N/A"
                semantic_rank = res["semantic_rank"] if res["semantic_rank"] != float("inf") else "N/A"
                print(
                    f"{i}. {res['title']}\n"
                    f"  RRF Score: {res['score']:.3f}\n"
                    f"  BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n"
                    f"  {res['description']}\n"
                )

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
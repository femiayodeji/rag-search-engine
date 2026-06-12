import argparse
from pathlib import Path

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.search_utils import GoldenDataset, GoldenTestCase, load_golden_dataset
from cli.lib.hybrid_search import HybridSearch
from cli.load_data import get_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

     # run evaluation logic here
    documents = get_movies()
    dataset: GoldenDataset = load_golden_dataset()
    for item in dataset.get("test_cases", []):
        query = item.get("query", "")
        relevant_docs = item.get("relevant_docs", [])

        search = HybridSearch(documents)

        # precision = relevant_retrieved / total_retrieved
        results = search.rrf_search(query, k=60, limit=limit)
        relevant_retrieved = sum(1 for result in results if result.get("title", "") in relevant_docs)
        total_retrieved = len(results)
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0

        # recall = relevant_retrieved / total_relevant
        total_relevant = len(relevant_docs)
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"k={limit}")
        print(f"- Query: {query}")
        print(f" - Precision@{limit}: {precision:.4f}")
        print(f" - Recall@{limit}: {recall:.4f}")
        print(f" - F1 Score: {f1:.4f}")
        print(f" - Retrieved: {', '.join(result.get('title', 'N/A') for result in results)}")
        print(f" - Relevant: {', '.join(relevant_docs)}\n")
        

    
if __name__ == "__main__":
    main()
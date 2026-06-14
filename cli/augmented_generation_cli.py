import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


from cli.lib.rag import get_rag_results
from cli.lib.hybrid_search import HybridSearch
from cli.load_data import get_movies

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = get_movies()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=5)
            print(f"Search Results:")
            for res in results:
                print(f"- {res.get('title', '')}")
            
            print(f"RAG Response:")
            rag_result = get_rag_results(query, results)
            print(rag_result)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
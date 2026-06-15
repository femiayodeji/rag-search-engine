import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


from cli.lib.rag import get_rag_results, get_rag_results_citation, get_rag_results_question_answering, get_rag_results_summary
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

    summary_parser = subparsers.add_parser(
        "summarize", help="Generate a search result summary using RAG"
    )
    summary_parser.add_argument("query", type=str, help="Search query for summary generation")
    summary_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for summary (default: 5)")

    citations_parser = subparsers.add_parser(
        "citations", help="Generate a search result summary with citations using RAG"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citation summary generation")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for citation summary (default: 5)")

    question_answering_parser = subparsers.add_parser(
        "question", help="Answer a question based on search results using RAG"
    )
    question_answering_parser.add_argument("question", type=str, help="Search query for question answering")
    question_answering_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use for question answering (default: 5)")


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

        case "summarize":
            query = args.query
            documents = get_movies()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=args.limit)
            print(f"Search Results:")
            for res in results:
                print(f"- {res.get('title', '')}")

            print(f"LLM Summary:")
            rag_summary = get_rag_results_summary(query, results)
            print(rag_summary)

        case "citations":
            query = args.query
            documents = get_movies()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=args.limit)
            print(f"Search Results:")
            for res in results:
                print(f"- {res.get('title', '')}")

            print(f"LLM Answer:")
            rag_citation_summary = get_rag_results_citation(query, results)
            print(rag_citation_summary)

        case "question":
            question = args.question
            documents = get_movies()
            search = HybridSearch(documents)
            results = search.rrf_search(question, limit=args.limit)
            print(f"Search Results:")
            for res in results:
                print(f"- {res.get('title', '')}")

            print(f"Answer:")
            rag_question_answering = get_rag_results_question_answering(question, results)
            print(rag_question_answering)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
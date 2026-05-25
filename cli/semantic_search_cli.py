#!/usr/bin/env python3
import sys
from pathlib import Path
from unittest import case


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_model, verify_embeddings
from cli.load_data import get_movies

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Load and verify the semantic model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_parser.add_argument("query_text", type=str, nargs='?', help="Text to generate embedding for")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a given query")
    embed_query_parser.add_argument("query_text", type=str, nargs='?', help="Query text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Load or create embeddings and verify their shape")

    search_parser = subparsers.add_parser("search", help="Search for relevant documents based on a query")
    search_parser.add_argument("query_text", type=str, nargs='?', help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a long text into smaller pieces")
    chunk_parser.add_argument("query_text", type=str, nargs='?', help="Long text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query_text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query_text)
        case "search":
            if not args.query_text:
                print("Error: query_text is required for the search command.")
                sys.exit(1)
            
            semantic_search = SemanticSearch()
            documents = get_movies()
            semantic_search.load_or_create_embeddings(documents)
            
            results = semantic_search.search(args.query_text, args.limit)
            
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"\t{result['description']}\n")

        case "chunk":
            if not args.query_text:
                print("Error: query_text is required for the chunk command.")
                sys.exit(1)
            characters = args.query_text.split(" ")
            n = args.chunk_size
            chunks = [characters[i:i + n] for i in range(0, len(characters), n)]
            print(f"Chunking {len(args.query_text)} characters")            
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}. {' '.join(chunk)}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
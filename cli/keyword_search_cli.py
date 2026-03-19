#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.inverted_index import InvertedIndex
from cli.load_data import get_movies
from cli.string_processor import process_str

def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")


    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()

        case "search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Cache files not found. Please build the index first.")
                exit(1)

            movies_result = []
            query_tokens =process_str(args.query)
            for query_token in query_tokens:
                query_doc_ids = index.get_documents(query_token)
                for query_doc_id in query_doc_ids:
                    movie = index.docmap[query_doc_id]
                    if movie not in movies_result:
                        movies_result.append(movie)
                        if len(movies_result) >= 5:
                             break
                if len(movies_result) >= 5:
                    break
                
            for movie in movies_result:
                print(f"{movie['id']}. {movie['title']}")

        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Cache files not found. Please build the index first.")
                exit(1)
            doc_id = args.doc_id
            term = args.term
            tf = index.get_tf(int(doc_id), term)
            print(tf)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
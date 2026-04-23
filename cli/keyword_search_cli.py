#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.inverted_index import InvertedIndex, bm25_idf_command
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

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a given document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")


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
            processed_terms = process_str(args.term)
            term = processed_terms[0] if processed_terms else args.term.lower()
            tf = index.get_tf(int(doc_id), term)
            print(tf)

        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Cache files not found. Please build the index first.")
                exit(1)
            processed_terms = process_str(args.term)
            term = processed_terms[0] if processed_terms else args.term.lower()
            document_ids = index.get_documents(term)
            term_match_doc_count = len(document_ids)
            total_doc_count = len(get_movies())
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Cache files not found. Please build the index first.")
                exit(1)
            doc_id = args.doc_id
            processed_terms = process_str(args.term)
            term = processed_terms[0] if processed_terms else args.term.lower()
            tf = index.get_tf(int(doc_id), term)
            document_ids = index.get_documents(term)
            term_match_doc_count = len(document_ids)
            total_doc_count = len(get_movies())
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")    
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
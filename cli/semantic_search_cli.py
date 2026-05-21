#!/usr/bin/env python3
import sys
from pathlib import Path
from unittest import case

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.lib.semantic_search import embed_text, verify_model, verify_embeddings

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Load and verify the semantic model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_parser.add_argument("query_text", type=str, nargs='?', help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Load or create embeddings and verify their shape")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query_text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
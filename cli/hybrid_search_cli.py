#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return
            min_score = min(scores)
            max_score = max(scores)
            if min_score == max_score:
                normalized = [1.0] * len(scores)
            else:
                normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
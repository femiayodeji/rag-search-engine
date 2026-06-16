import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


from cli.lib.multimodal_search import MultimodalSearch, verify_image_embedding

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_embedding_parser.add_argument("image", type=str, help="Path to image for embedding verification")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
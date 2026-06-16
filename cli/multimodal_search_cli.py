import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


from cli.lib.multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_embedding_parser.add_argument("image", type=str, help="Path to image for embedding verification")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies by image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to image for search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for i, result in enumerate(results, start=1):
                description = result.get('description', '')
                print(f"{i}. {result.get('title', '')} (similarity: {result.get('similarity', 0):.3f})")
                print(f"   {description[:100]}{'...' if len(description) > 100 else ''}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
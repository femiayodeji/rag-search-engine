import mimetypes
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.llm_utils import llm_request_parts
from google.genai import types

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--query", type=str, help="Search query for multimodal search")
    parser.add_argument("--image", type=str, help="Path to image for multimodal search")
    

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image_data = f.read()

    query = args.query.strip()

    system_propmt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    parts = [
        system_propmt,
        types.Part.from_bytes(data=image_data, mime_type=mime),
        query,
    ]

    llm_response = llm_request_parts(parts)

    print(f"Rewritten query: {llm_response.text.strip()}")
    if llm_response.usage_metadata is not None:
        print(f"Total tokens:    {llm_response.usage_metadata.total_token_count}")
        

if __name__ == "__main__":
    main()
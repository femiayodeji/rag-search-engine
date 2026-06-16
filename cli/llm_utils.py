import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai
from google.genai.types import GenerateContentResponse

client = genai.Client(api_key=api_key)

def llm_request(query: str, model="gemma-4-31b-it") -> GenerateContentResponse:
    try:
       response: GenerateContentResponse = client.models.generate_content(model=model, contents=query)
    except Exception as e:
        print(f"LLM request failed: {e}")
        raise RuntimeError("LLM request failed") from e
    return response


def llm_request_parts(parts: list, model="gemma-4-31b-it"):
    try:
       response: GenerateContentResponse = client.models.generate_content(model=model, contents=parts)
    except Exception as e:
        print(f"LLM request failed: {e}")
        raise RuntimeError("LLM request failed") from e
    return response

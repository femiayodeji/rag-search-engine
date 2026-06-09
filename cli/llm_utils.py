import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai
from google.genai.types import GenerateContentResponse

client = genai.Client(api_key=api_key)

client_response: GenerateContentResponse = client.models.generate_content(model="gemma-4-31b-it", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

def llm_request(query: str, model="gemma-4-31b-it") -> GenerateContentResponse:
    response: GenerateContentResponse = client.models.generate_content(model=model, contents=query)
    return response
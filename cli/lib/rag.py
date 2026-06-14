from cli.constants import get_rag_prompt
from cli.llm_utils import llm_request
from cli.search_utils import SearchResult


def get_rag_results(query: str, search_result: list[SearchResult]) -> str:
    docs = "\n\n".join(
        f"Title: {res.get('title', '')}\nDescription: {res.get('description', '')}\nDocument: {res.get('document', '')}"
        for res in search_result
    )
    prompt = get_rag_prompt(query, docs)
    try:
        llm_response = llm_request(prompt)
        return llm_response.text or ""
    except Exception as e:
        print(f"Error during LLM request: {e}")
        return ""

from cli.constants import get_citation_summary_prompt, get_rag_prompt, get_summary_prompt
from cli.llm_utils import llm_request
from cli.search_utils import SearchResult

def results_to_formatted_strings(results: list[SearchResult]) -> str:
    return  "\n\n".join(
        f"Title: {res.get('title', '')}\nDescription: {res.get('description', '')}\nDocument: {res.get('document', '')}"
        for res in results
    )


def get_rag_results(query: str, search_result: list[SearchResult]) -> str:
    docs = results_to_formatted_strings(search_result)
    prompt = get_rag_prompt(query, docs)
    return get_rag(prompt)

def get_rag_results_summary(query: str, search_result: list[SearchResult]) -> str:
    docs = results_to_formatted_strings(search_result)
    prompt = get_summary_prompt(query, docs)
    return get_rag(prompt)

def get_rag_results_citation(query: str, search_result: list[SearchResult]) -> str:
    docs = results_to_formatted_strings(search_result)
    prompt = get_citation_summary_prompt(query, docs)
    return get_rag(prompt)

def get_rag(prompt: str) -> str:
    try:
        llm_response = llm_request(prompt)
        return llm_response.text or ""
    except Exception as e:
        print(f"Error during LLM request: {e}")
        return ""

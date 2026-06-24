# RAG Search Engine

A search engine built over a movie dataset that explores and combines multiple retrieval techniques — keyword search, semantic search, hybrid search, and retrieval-augmented generation (RAG). It is built primarily as a learning project to understand how these systems work from the ground up.

## What it does

The project lets you query a corpus of movie descriptions using several different search strategies:

- **Keyword search** — BM25-based inverted index with TF-IDF scoring
- **Semantic search** — dense vector embeddings using `sentence-transformers` (`All-MiniLM-L6-v2`), with support for chunk-level embeddings that aggregate back to document-level scores
- **Hybrid search** — combines BM25 and semantic scores either via weighted fusion (alpha parameter) or Reciprocal Rank Fusion (RRF)
- **Query enhancement** — spell correction, query rewriting, and query expansion powered by an LLM before retrieval
- **Reranking** — post-retrieval reranking using an LLM (individual or batch) or a cross-encoder model
- **RAG** — uses hybrid search results as context for a Gemini LLM to answer questions, produce summaries, and generate cited responses
- **Multimodal search** — image-to-movie search using image embeddings

## Key concepts

**BM25** ranks documents by term frequency and inverse document frequency, penalising very long documents. It handles exact keyword matches well but misses synonyms and paraphrase.

**Semantic search** encodes both documents and queries into a shared embedding space. Cosine similarity finds conceptually related results even when no keywords overlap. Documents are also chunked before embedding so that a single relevant passage can surface a long document.

**Hybrid search with RRF** avoids the need to tune score normalisation by merging ranked lists instead of raw scores. Each document gets a score of `1 / (k + rank)` from both lists; the combined scores are then sorted. This tends to be more robust than weighted fusion.

**RAG** feeds the top retrieved documents into an LLM prompt as context. The LLM then generates a grounded answer, summary, or cited response rather than hallucinating from parametric memory alone.

**Evaluation** measures retrieval quality with Precision@k, Recall@k, and F1 against a golden dataset of queries with known relevant documents.

## Setup

**Requirements:** Python 3.13+, a Gemini API key.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Set your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Running the CLIs

Each CLI can be run directly. Pass `--help` to any command to see available subcommands.

```bash
# Build the inverted index
uv run cli/keyword_search_cli.py build

# Keyword search
uv run cli/keyword_search_cli.py bm25search "action movie in space"

# Semantic search
uv run cli/semantic_search_cli.py search "lonely astronaut on mars" --limit 5

# Hybrid search with RRF
uv run cli/hybrid_search_cli.py rrf-search "crime thriller in New York"

# Hybrid search with query expansion and LLM reranking
uv run cli/hybrid_search_cli.py rrf-search "gangster film" --enhance expand --rerank-method batch

# RAG — answer a question using retrieved context
uv run cli/augmented_generation_cli.py question "What movies involve a heist gone wrong?"

# RAG — summarise results
uv run cli/augmented_generation_cli.py summarize "space exploration films"

# RAG — answer with citations
uv run cli/augmented_generation_cli.py citations "films about redemption"

# Multimodal image search
uv run cli/multimodal_search_cli.py image_search path/to/image.jpg

# Evaluation
uv run cli/evaluation_cli.py --limit 5
```

## Project structure

```
cli/
  keyword_search_cli.py       BM25 / TF-IDF keyword search
  semantic_search_cli.py      Dense embedding search and chunking
  hybrid_search_cli.py        Weighted and RRF hybrid search with query enhancement and reranking
  augmented_generation_cli.py RAG — QA, summarisation, citations
  multimodal_search_cli.py    Image-based search
  evaluation_cli.py           Precision/Recall/F1 evaluation against a golden dataset
  inverted_index.py           BM25 inverted index implementation
  lib/
    semantic_search.py        Embedding generation and cosine similarity search
    chunk_semantic_search.py  Chunk-level embeddings with document-level score aggregation
    hybrid_search.py          Hybrid search, RRF, reranking, query enhancement
    rag.py                    RAG prompt construction and LLM calls
    multimodal_search.py      Image embedding and search
data/
  movies.json                 Movie corpus
  golden_dataset.json         Evaluation dataset with queries and relevant documents
cache/
  movie_embeddings.npy        Pre-computed document embeddings
  chunk_embeddings.npy        Pre-computed chunk embeddings
  chunk_metadata.json         Chunk-to-document mapping
```

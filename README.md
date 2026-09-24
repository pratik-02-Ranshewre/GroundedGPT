# GroundedGPT

A fully local, zero-cost Retrieval-Augmented Generation (RAG) application that lets you ask questions and generate quizzes from your own documents (text files and PDFs) — grounded entirely in your uploaded content, with no API keys, no subscriptions, and no external cloud dependency.

## What it does

- **Ask a Question** — upload your own notes/PDFs and ask natural-language questions. Answers are generated using only the retrieved content from your documents, not the LLM's general training knowledge.
- **Generate Quiz** — enter a topic and get auto-generated multiple-choice questions, sourced only from your uploaded material, returned as structured JSON and rendered as an interactive quiz.
- **Multi-format ingestion** — supports both `.txt` and `.pdf` files, loaded automatically from a single `Data/` folder.

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| LLM (local, no API) | Ollama (Llama 3) |
| Embeddings (local) | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | Chroma |
| PDF parsing | PyMuPDF |
| UI | Streamlit |

Everything runs on your own machine — no OpenAI/Gemini/Mistral API keys required, no per-query cost.

## How it works

1. **Ingestion** — documents from `Data/` (`.txt` and `.pdf`) are loaded and split into overlapping chunks (500 characters, 50-character overlap) so that ideas aren't severed at chunk boundaries.
2. **Embedding** — each chunk is converted into a 384-dimensional vector locally using `all-MiniLM-L6-v2`, capturing semantic meaning rather than exact keywords.
3. **Storage & Retrieval** — vectors are stored in a local Chroma database. A user's question is embedded the same way and compared against all stored chunks to retrieve the most semantically relevant ones.
4. **Generation** — the retrieved chunks are inserted into a prompt alongside the user's question, and Llama 3 (via Ollama) generates an answer grounded in that retrieved context — explicitly instructed to say "I don't know" rather than hallucinate when the context doesn't contain the answer.
5. **Quiz mode** reuses the same retrieval step, but instead prompts the LLM to generate multiple-choice questions as JSON, which is parsed (with a regex fallback for cases where the LLM adds extra commentary around the JSON) and rendered as an interactive quiz.

## Setup

### 1. Install Ollama and pull a model
```bash
# Download from https://ollama.com/download, then:
ollama pull llama3
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
```

### 3. Install dependencies
```bash
pip install streamlit langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama langchain-text-splitters sentence-transformers chromadb pymupdf
```

### 4. Add your documents
Place `.txt` and/or `.pdf` files inside a `Data/` folder in the project root.

### 5. Run the app
```bash
streamlit run app.py
```

## Design decisions worth noting

- **Chunk overlap (50 chars)** prevents a sentence or idea from being split across two chunks in a way that loses meaning at retrieval time.
- **Grounding instructions in the prompt** — the LLM is explicitly told to answer only from the provided context and to say "I don't know based on the provided notes" otherwise, reducing hallucination risk compared to an ungrounded LLM call.
- **JSON extraction fallback** — LLM output isn't always perfectly clean JSON (it sometimes adds a sentence of commentary before/after). A regex-based extractor pulls out just the JSON array before parsing, so quiz generation is resilient to this common failure mode.
- **Fully local** — chosen specifically to avoid any per-query API cost or external data dependency; trade-off is slower generation on CPU-only machines compared to a hosted API.

## Possible extensions

- Similarity-score threshold check before calling the LLM (skip generation entirely if nothing relevant was retrieved, rather than relying on the LLM to notice)
- Flashcard generation mode (same pattern as quiz generation)
- Support for additional file types (`.docx`, `.md`)
- Incremental re-indexing instead of rebuilding the vector store from scratch on every run

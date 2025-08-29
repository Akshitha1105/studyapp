# RAG Study App

A Streamlit-based Study App with document uploads, FAISS vector search, SentenceTransformers embeddings, and a RAG pipeline powered by Hugging Face Inference API (default) or optional local transformers.

## Features
- Upload .pdf, .docx, .txt, .md
- Text extraction, chunking, and persistent FAISS index in `./kb_store`
- Retrieve top-k chunks and generate grounded answers with citations
- Summarizer, Flashcard generator, Quiz maker
- Prompt Builder to generate a Cursor prompt reflecting current app + KB state

## Requirements
- Python 3.10+
- `HUGGINGFACEHUB_API_TOKEN` set in your environment for remote inference

## Setup
```bash
# (Preferred) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Export your HF token
export HUGGINGFACEHUB_API_TOKEN=hf_...
```

## Run
```bash
streamlit run app.py
```

## Notes
- FAISS index and metadata are stored in `./kb_store` and persist across sessions.
- Use the sidebar Reset to wipe the knowledge base.
- Local transformers option may require a GPU or a CPU-friendly model.
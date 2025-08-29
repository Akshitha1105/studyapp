from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def build_prompt_builder_text(store) -> str:
	stats = {
		"chunks": len(store.metadata),
		"index_size": getattr(store.index, "ntotal", 0) if store.index is not None else 0,
	}
	lines = [
		"You are Cursor building on a Streamlit RAG Study App.",
		"- UI: Streamlit with sidebar (upload, settings, reset) and tabs (Chat, Summarizer, Flashcards, Quiz, Prompt Builder).",
		"- Embeddings: sentence-transformers/all-MiniLM-L6-v2; Vector store: FAISS (IP cosine).",
		"- RAG: retrieve top-k chunks; LLM: Hugging Face Inference API mistralai/Mistral-7B-Instruct-v0.3 (or local).",
		f"- Knowledge base stats: chunks={stats['chunks']}, index_size={stats['index_size']}.",
	]
	return "\n".join(lines)
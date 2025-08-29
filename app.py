import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from rag.extractors import extract_text_with_metadata
from rag.chunking import chunk_text
from rag.embeddings import EmbeddingModel
from rag.store import FaissVectorStore
from rag.pipeline import RagPipeline
from rag.utils import ensure_dir, build_prompt_builder_text

KB_DIR = Path("./kb_store")
INDEX_PATH = KB_DIR / "faiss.index"
META_PATH = KB_DIR / "metadata.jsonl"


def init_state():
	if "vector_store" not in st.session_state:
		ensure_dir(KB_DIR)
		st.session_state.vector_store = FaissVectorStore(str(INDEX_PATH), str(META_PATH))
		st.session_state.vector_store.load_if_exists()
	if "embedder" not in st.session_state:
		st.session_state.embedder = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
	if "pipeline" not in st.session_state:
		st.session_state.pipeline = RagPipeline(st.session_state.embedder, st.session_state.vector_store)
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []


def reset_kb():
	if KB_DIR.exists():
		shutil.rmtree(KB_DIR)
	ensure_dir(KB_DIR)
	st.session_state.vector_store = FaissVectorStore(str(INDEX_PATH), str(META_PATH))
	st.session_state.pipeline = RagPipeline(st.session_state.embedder, st.session_state.vector_store)
	st.toast("Knowledge base reset.")


def sidebar():
	st.sidebar.title("Settings")
	st.sidebar.write("Upload study documents and tune retrieval.")
	with st.sidebar.expander("Upload Documents", expanded=True):
		uploaded_files = st.file_uploader(
			"Add .pdf, .docx, .txt, .md files",
			type=["pdf", "docx", "txt", "md"],
			accept_multiple_files=True,
			key="uploader",
		)
		if uploaded_files and st.button("Ingest Documents", type="primary"):
			with st.spinner("Processing documents..."):
				all_records: List[Dict[str, Any]] = []
				for file in uploaded_files:
					records = extract_text_with_metadata(file)
					for rec in records:
						chunks = chunk_text(rec["text"], chunk_size=800, overlap=120)
						for idx, ch in enumerate(chunks):
							meta = {k: v for k, v in rec.items() if k != "text"}
							meta.update({
								"chunk_id": idx,
								"text": ch,
								"preview": ch[:300],
							})
							all_records.append({"text": ch, "metadata": meta})
				texts = [r["text"] for r in all_records]
				metas = [r["metadata"] for r in all_records]
				st.session_state.pipeline.index_texts(texts, metas)
				st.success(f"Ingested {len(all_records)} chunks from {len(uploaded_files)} files.")
	with st.sidebar.expander("Retrieval Settings", expanded=False):
		st.session_state.top_k = st.number_input("Top-k Chunks", min_value=1, max_value=20, value=5)
		st.session_state.temperature = st.slider("Temperature (if local)", 0.0, 1.5, 0.3, 0.1)
		st.session_state.use_local = st.toggle("Use Local Model (transformers)", value=False)
		st.session_state.local_model = st.text_input("Local Model ID", value="mistralai/Mistral-7B-Instruct-v0.3")
		st.session_state.hf_model = st.text_input("HF Inference Model", value="mistralai/Mistral-7B-Instruct-v0.3")
		st.caption("Set HUGGINGFACEHUB_API_TOKEN in environment for HF Inference API.")
	with st.sidebar.expander("Maintenance", expanded=False):
		if st.button("Reset Knowledge Base", type="secondary"):
			reset_kb()


def main_ui():
	st.title("RAG Study App 📚")
	tabs = st.tabs(["Chat", "Summarizer", "Flashcards", "Quiz", "Prompt Builder"]) 
	# Chat
	with tabs[0]:
		user_query = st.text_input("Ask a question about your documents")
		if st.button("Ask") and user_query.strip():
			with st.spinner("Retrieving and generating answer..."):
				answer, citations = st.session_state.pipeline.answer_query(
					user_query,
					top_k=int(st.session_state.get("top_k", 5)),
					use_local=bool(st.session_state.get("use_local", False)),
					model_id=str(st.session_state.get("local_model" if st.session_state.get("use_local", False) else "hf_model")),
					temperature=float(st.session_state.get("temperature", 0.3)),
				)
				st.write(answer)
				if citations:
					st.markdown("**Citations:**")
					for cit in citations:
						st.caption(f"- {cit}")
	# Summarizer
	with tabs[1]:
		if st.button("Summarize Knowledge Base"):
			with st.spinner("Summarizing..."):
				summary = st.session_state.pipeline.summarize_corpus(
					use_local=bool(st.session_state.get("use_local", False)),
					model_id=str(st.session_state.get("local_model" if st.session_state.get("use_local", False) else "hf_model")),
				)
				st.write(summary)
	# Flashcards
	with tabs[2]:
		if st.button("Generate Flashcards"):
			with st.spinner("Generating flashcards..."):
				cards = st.session_state.pipeline.generate_flashcards(
					use_local=bool(st.session_state.get("use_local", False)),
					model_id=str(st.session_state.get("local_model" if st.session_state.get("use_local", False) else "hf_model")),
				)
				for i, c in enumerate(cards, 1):
					st.markdown(f"**Q{i}.** {c['question']}")
					st.caption(f"Answer: {c['answer']}")
	# Quiz
	with tabs[3]:
		if st.button("Generate Quiz"):
			with st.spinner("Generating quiz..."):
				quiz = st.session_state.pipeline.generate_quiz(
					use_local=bool(st.session_state.get("use_local", False)),
					model_id=str(st.session_state.get("local_model" if st.session_state.get("use_local", False) else "hf_model")),
				)
				for i, q in enumerate(quiz, 1):
					st.markdown(f"**Q{i}.** {q['question']}")
					for opt in q["options"]:
						st.write(f"- {opt}")
					st.caption(f"Correct: {q['answer']}")
	# Prompt Builder
	with tabs[4]:
		if st.button("Build Cursor Prompt"):
			prompt_text = build_prompt_builder_text(st.session_state.vector_store)
			st.text_area("Generated Prompt", prompt_text, height=300)


def main():
	init_state()
	sidebar()
	main_ui()


if __name__ == "__main__":
	main()
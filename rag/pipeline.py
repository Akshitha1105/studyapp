from __future__ import annotations

from typing import List, Dict, Any, Tuple

import numpy as np

from .embeddings import EmbeddingModel
from .store import FaissVectorStore
from .llm import HFInferenceClient, LocalTransformersClient


ANSWER_SYSTEM = (
	"You are a helpful study assistant. Answer using ONLY the provided context. "
	"Cite sources as (source p. page, chunk chunk_id). If uncertain, say you don't know."
)

SUMMARY_SYSTEM = "You are a concise summarizer. Create a brief, faithful summary of the content."
FLASHCARD_SYSTEM = "Create compact Q&A flashcards that test key concepts. Return JSON with question and answer."
QUIZ_SYSTEM = "Create multiple-choice questions with 4 options and 1 correct answer. Return JSON."


class RagPipeline:
	def __init__(self, embedder: EmbeddingModel, store: FaissVectorStore) -> None:
		self.embedder = embedder
		self.store = store

	def index_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
		embeddings = self.embedder.embed(texts)
		self.store.add(embeddings, metadatas)

	def _choose_llm(self, use_local: bool, model_id: str):
		return LocalTransformersClient(model_id) if use_local else HFInferenceClient(model_id)

	def _format_context(self, docs: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
		lines = []
		citations = []
		for d in docs:
			meta = d["metadata"]
			context_line = meta.get("text", "") if isinstance(meta, dict) else ""
			# In our design, text isn't in metadata; include no text here and rely on later aggregation
		lines.append(meta.get("preview", ""))
		# Build citation string
		source = meta.get("source")
		page = meta.get("page")
		chunk_id = meta.get("chunk_id")
		citations.append(f"{source} p.{page} chunk {chunk_id}")
		return "\n\n".join(lines), citations

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		q_emb = self.embedder.embed([query])
		results = self.store.search(q_emb, top_k=top_k)[0]
		return results

	def answer_query(self, query: str, top_k: int = 5, use_local: bool = False, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", temperature: float = 0.3) -> Tuple[str, List[str]]:
		results = self.retrieve(query, top_k=top_k)
		# Build context as concatenated chunks from metadata preview field; add fallback
		context_blocks = []
		citations = []
		for r in results:
			m = r["metadata"]
			context_blocks.append(m.get("text", m.get("preview", "")))
			source = m.get("source")
			page = m.get("page")
			chunk_id = m.get("chunk_id")
			citations.append(f"{source} p.{page} chunk {chunk_id}")
		context = "\n---\n".join([c for c in context_blocks if c])
		if not context:
			context = "No relevant context found."
		user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
		client = self._choose_llm(use_local, model_id)
		answer = client.complete(ANSWER_SYSTEM, user_prompt, temperature=temperature)
		return answer, citations

	def summarize_corpus(self, use_local: bool = False, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> str:
		# Use a subset to avoid overlong context
		texts = []
		for m in self.store.metadata[:50]:
			texts.append(m.get("text", m.get("preview", "")))
		context = "\n".join([t for t in texts if t])
		client = self._choose_llm(use_local, model_id)
		return client.complete(SUMMARY_SYSTEM, context)

	def generate_flashcards(self, use_local: bool = False, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> List[Dict[str, str]]:
		texts = []
		for m in self.store.metadata[:40]:
			texts.append(m.get("text", m.get("preview", "")))
		context = "\n".join([t for t in texts if t])
		client = self._choose_llm(use_local, model_id)
		raw = client.complete(FLASHCARD_SYSTEM, context)
		# Best-effort parse
		import json
		cards: List[Dict[str, str]] = []
		try:
			data = json.loads(raw)
			if isinstance(data, list):
				for item in data:
					q = item.get("question")
					a = item.get("answer")
					if q and a:
						cards.append({"question": q, "answer": a})
		except Exception:
			# Fallback naive split
			for line in raw.splitlines():
				if ":" in line:
					q, a = line.split(":", 1)
					cards.append({"question": q.strip(), "answer": a.strip()})
		return cards[:10]

	def generate_quiz(self, use_local: bool = False, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> List[Dict[str, Any]]:
		texts = []
		for m in self.store.metadata[:40]:
			texts.append(m.get("text", m.get("preview", "")))
		context = "\n".join([t for t in texts if t])
		client = self._choose_llm(use_local, model_id)
		raw = client.complete(QUIZ_SYSTEM, context)
		import json
		quiz: List[Dict[str, Any]] = []
		try:
			data = json.loads(raw)
			if isinstance(data, list):
				for item in data:
					q = item.get("question")
					opts = item.get("options") or []
					ans = item.get("answer")
					if q and opts and ans:
						quiz.append({"question": q, "options": opts[:4], "answer": ans})
		except Exception:
			# Fallback produce one simple question from raw
			quiz.append({"question": raw[:120] + "?", "options": ["A", "B", "C", "D"], "answer": "A"})
		return quiz[:5]
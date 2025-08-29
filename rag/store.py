from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np


class FaissVectorStore:
	"""FAISS index with cosine similarity and persistent metadata JSONL."""

	def __init__(self, index_path: str, meta_path: str) -> None:
		self.index_path = Path(index_path)
		self.meta_path = Path(meta_path)
		self.index: Optional[faiss.IndexFlatIP] = None
		self.metadata: List[Dict[str, Any]] = []

	def _create_index(self, dim: int) -> None:
		self.index = faiss.IndexFlatIP(dim)

	def load_if_exists(self) -> None:
		if self.index_path.exists() and self.meta_path.exists():
			# Load metadata
			self.metadata = []
			with self.meta_path.open("r", encoding="utf-8") as f:
				for line in f:
					self.metadata.append(json.loads(line))
			# Load index
			self.index = faiss.read_index(str(self.index_path))

	def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
		if self.index is None:
			self._create_index(embeddings.shape[1])
		# FAISS IP assumes vectors are normalized for cosine similarity
		self.index.add(embeddings.astype(np.float32))
		with self.meta_path.open("a", encoding="utf-8") as f:
			for meta in metadatas:
				f.write(json.dumps(meta, ensure_ascii=False) + "\n")
		self.metadata.extend(metadatas)
		faiss.write_index(self.index, str(self.index_path))

	def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Dict[str, Any]]]:
		if self.index is None or self.index.ntotal == 0:
			return [[] for _ in range(query_embeddings.shape[0])]
		dists, indices = self.index.search(query_embeddings.astype(np.float32), top_k)
		results: List[List[Dict[str, Any]]] = []
		for row_indices, row_scores in zip(indices, dists):
			hits: List[Dict[str, Any]] = []
			for idx, score in zip(row_indices, row_scores):
				if idx == -1:
					continue
				meta = self.metadata[idx]
				hits.append({"score": float(score), "metadata": meta})
			results.append(hits)
		return results

	def wipe(self) -> None:
		self.index = None
		self.metadata = []
		if self.index_path.exists():
			self.index_path.unlink()
		if self.meta_path.exists():
			self.meta_path.unlink()
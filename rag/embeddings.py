from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
	"""Wrapper around SentenceTransformer embeddings with lazy loading."""

	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
		self.model_name = model_name
		self._model: SentenceTransformer | None = None

	def _load(self) -> SentenceTransformer:
		if self._model is None:
			self._model = SentenceTransformer(self.model_name)
		return self._model

	def embed(self, texts: List[str]) -> np.ndarray:
		model = self._load()
		embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
		return embeddings
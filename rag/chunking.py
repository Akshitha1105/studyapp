from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
	"""Simple word-based chunking with overlap to preserve context."""
	words = text.split()
	if not words:
		return []
	chunks: List[str] = []
	start = 0
	while start < len(words):
		end = min(start + chunk_size, len(words))
		chunk = " ".join(words[start:end])
		chunks.append(chunk)
		if end == len(words):
			break
		start = max(0, end - overlap)
	return chunks
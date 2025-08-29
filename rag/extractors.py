from __future__ import annotations

from typing import List, Dict, Any

import pdfplumber
from docx import Document


def _extract_from_pdf(file_like) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with pdfplumber.open(file_like) as pdf:
		for page_num, page in enumerate(pdf.pages, start=1):
			text = page.extract_text() or ""
			if text.strip():
				records.append({
					"text": text,
					"source": getattr(file_like, 'name', 'uploaded.pdf'),
					"file_type": "pdf",
					"page": page_num,
				})
	return records


def _extract_from_docx(file_like) -> List[Dict[str, Any]]:
	doc = Document(file_like)
	paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
	text = "\n".join(paras)
	return [{
		"text": text,
		"source": getattr(file_like, 'name', 'uploaded.docx'),
		"file_type": "docx",
		"page": None,
	}]


def _extract_from_txt(file_like) -> List[Dict[str, Any]]:
	text = file_like.read()
	if isinstance(text, bytes):
		text = text.decode("utf-8", errors="ignore")
	return [{
		"text": text,
		"source": getattr(file_like, 'name', 'uploaded.txt'),
		"file_type": "txt",
		"page": None,
	}]


def _extract_from_md(file_like) -> List[Dict[str, Any]]:
	# Treat markdown as plain text; renderers are out of scope
	return _extract_from_txt(file_like)


def extract_text_with_metadata(file_like) -> List[Dict[str, Any]]:
	"""Extract text blocks and attach metadata. Supports pdf, docx, txt, md."""
	name: str = getattr(file_like, "name", "uploaded")
	lower = name.lower()
	# Streamlit uploads are BytesIO-like and can be re-used by libraries directly.
	if lower.endswith(".pdf"):
		return _extract_from_pdf(file_like)
	if lower.endswith(".docx"):
		return _extract_from_docx(file_like)
	if lower.endswith(".txt"):
		return _extract_from_txt(file_like)
	if lower.endswith(".md"):
		return _extract_from_md(file_like)
	raise ValueError(f"Unsupported file type for {name}")
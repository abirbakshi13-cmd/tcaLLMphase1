from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedSnippet:
    doc_id: str
    snippet: str


@dataclass
class VectorStore:
    model_name: str
    embeddings: List[List[float]]
    documents: List[RetrievedSnippet]


def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def iter_pdf_files(data_dir: str) -> Iterable[str]:
    for name in os.listdir(data_dir):
        if name.lower().endswith(".pdf"):
            yield os.path.join(data_dir, name)


def build_vector_store(
    data_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 800,
    overlap: int = 120,
) -> VectorStore:
    model = SentenceTransformer(model_name)
    embeddings: List[List[float]] = []
    documents: List[RetrievedSnippet] = []

    for file_path in iter_pdf_files(data_dir):
        doc_id = os.path.basename(file_path)
        text = load_pdf_text(file_path)
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            documents.append(RetrievedSnippet(doc_id=doc_id, snippet=chunk))

    if documents:
        vectors = model.encode(
            [item.snippet for item in documents],
            normalize_embeddings=True,
        )
        embeddings = vectors.tolist()

    return VectorStore(model_name=model_name, embeddings=embeddings, documents=documents)


def _cosine_similarity(query_vec: Sequence[float], doc_vec: Sequence[float]) -> float:
    return float(sum(q * d for q, d in zip(query_vec, doc_vec)))


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 3,
) -> List[RetrievedSnippet]:
    if not store.documents:
        return []

    model = SentenceTransformer(store.model_name)
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    scored: List[Tuple[float, RetrievedSnippet]] = []
    for vector, doc in zip(store.embeddings, store.documents):
        score = _cosine_similarity(query_vec, vector)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

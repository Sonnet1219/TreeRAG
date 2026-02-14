"""Chunking logic for node content."""

from __future__ import annotations


def chunk_content(
    content: str,
    chunk_size: int = 200,
    overlap: int = 50,
    min_chars: int = 20,
) -> list[str]:
    if chunk_size <= 0:
        return []

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    chunks: list[str] = []
    paragraphs = [para.strip() for para in content.split("\n\n")]
    for paragraph in paragraphs:
        if len(paragraph) < min_chars:
            continue

        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            piece = paragraph[start:end].strip()
            if len(piece) >= min_chars:
                chunks.append(piece)
            start += step

    return chunks

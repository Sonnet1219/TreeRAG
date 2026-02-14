"""Step 3: answer synthesis from retrieved evidence."""

from __future__ import annotations

from tree_rag.types import RagConfig, RetrievedChunk
from tree_rag.utils.openai_client import OpenAICompatibleClient


SYNTHESIS_PROMPT = """Based on the evidence retrieved from the document, answer the user question.

Rules:
1. Answer using only the provided evidence. Do not invent facts.
2. Cite sources after key claims using this format: [Source: section path].
3. If evidence is insufficient, explicitly state the limitation.
4. Keep the answer concise and accurate.

User Question: {query}

Retrieved Evidence:
{context}

Answer:
"""


def build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, item in enumerate(retrieved_chunks, start=1):
        parts.append(
            f"[Evidence{index}] Source: {item.chunk.heading_path}\n{item.chunk.text}"
        )
    return "\n\n".join(parts)


def synthesize(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    config: RagConfig,
    client: OpenAICompatibleClient | None,
    mock: bool,
) -> str:
    if not retrieved_chunks:
        return "Insufficient evidence retrieved to answer this question."

    if mock or client is None:
        lines = ["Based on retrieved evidence:"]
        for idx, item in enumerate(retrieved_chunks, start=1):
            lines.append(
                f"[{idx}] (Source: {item.chunk.heading_path}) \"{item.chunk.text}\""
            )
        return "\n".join(lines)

    context = build_context(retrieved_chunks)
    prompt = SYNTHESIS_PROMPT.format(query=query, context=context)
    return client.chat_completion(
        model=config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous document QA assistant. Cite evidence strictly.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=768,
        temperature=0.2,
    )

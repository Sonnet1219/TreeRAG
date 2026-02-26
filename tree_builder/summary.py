"""Summary generation for document trees."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import error as urlerror
from urllib import request

from tree_builder.tree import DocumentTree, postorder_nodes


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


@dataclass
class MockSummarizer:
    """Simple truncation-based summarizer for local testing."""

    max_chars: int = 100
    empty_content_placeholder: str = "No content available."
    empty_children_placeholder: str = "No child summaries available."

    def summarize_leaf(self, heading: str, content: str) -> str:
        normalized = _normalize_space(content)
        if not normalized:
            return self.empty_content_placeholder
        return normalized[: self.max_chars]

    def summarize_parent(
        self,
        heading: str,
        children_summaries: list[str],
        own_content: str = "",
    ) -> str:
        own = _normalize_space(own_content)
        children = _normalize_space(" ".join(children_summaries))

        if own and children:
            merged = f"{own} {children}"
            return merged[: self.max_chars]
        if own:
            return own[: self.max_chars]
        if children:
            return children[: self.max_chars]
        return self.empty_children_placeholder


class Summarizer(Protocol):
    def summarize_leaf(self, heading: str, content: str) -> str:
        """Generate one short summary for a leaf section."""

    def summarize_parent(
        self,
        heading: str,
        children_summaries: list[str],
        own_content: str = "",
    ) -> str:
        """Generate one short summary for a parent section."""


@dataclass
class LLMSummarizerStub:
    """Placeholder that reserves LLM mode but is not implemented in this phase."""

    provider: str

    def summarize_leaf(self, heading: str, content: str) -> str:
        raise NotImplementedError(
            f"LLM summarizer for provider '{self.provider}' is not implemented in this phase."
        )

    def summarize_parent(
        self,
        heading: str,
        children_summaries: list[str],
        own_content: str = "",
    ) -> str:
        raise NotImplementedError(
            f"LLM summarizer for provider '{self.provider}' is not implemented in this phase."
        )


@dataclass
class OpenAICompatibleSummarizer:
    """OpenAI-compatible chat-completions summarizer."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    max_tokens: int = 250
    temperature: float = 0.2
    max_content_chars: int = 2000

    def summarize_leaf(self, heading: str, content: str) -> str:
        snippet = _normalize_space(content[:self.max_content_chars])
        if not snippet:
            snippet = "No content available."
        prompt = (
            "Summarize this section in 2-3 concise sentences, covering the main points.\n"
            f"Heading: {heading}\n"
            f"Content: {snippet}"
        )
        return self._chat_completion(prompt)

    def summarize_parent(
        self,
        heading: str,
        children_summaries: list[str],
        own_content: str = "",
    ) -> str:
        own = _normalize_space(own_content[:self.max_content_chars])

        per_child_limit = max(200, self.max_content_chars // max(len(children_summaries), 1))
        formatted_children = "\n".join(
            f"  - {s[:per_child_limit]}" for s in children_summaries
        ) if children_summaries else ""

        prompt = (
            "Summarize this section in 2-3 concise sentences based on its subsections.\n"
            f"Heading: {heading}\n"
        )
        if own:
            prompt += f"Section overview: {own}\n"
        if formatted_children:
            prompt += f"Subsections:\n{formatted_children}\n"
        if not own and not formatted_children:
            prompt += "No content available.\n"

        return self._chat_completion(prompt)

    def _chat_completion(self, user_prompt: str) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You summarize markdown sections. Return plain text only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"OpenAI request failed with status {exc.code}: {details[:300]}"
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(response_body)
            content = parsed["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI response format is invalid.") from exc

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        normalized = _normalize_space(str(content))
        if not normalized:
            raise RuntimeError("OpenAI returned an empty summary.")
        return normalized


def build_llm_summarizer_from_env(provider: str) -> Summarizer:
    provider_normalized = provider.lower().strip()
    if provider_normalized != "openai":
        return LLMSummarizerStub(provider=provider_normalized)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for --mode llm with --provider openai.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "250"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_content_chars = int(os.getenv("OPENAI_MAX_CONTENT_CHARS", "2000"))

    return OpenAICompatibleSummarizer(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        max_content_chars=max_content_chars,
    )


def generate_summaries(tree: DocumentTree, summarizer: Summarizer) -> None:
    """Generate summaries bottom-up for all non-root nodes."""
    for node in postorder_nodes(tree.root):
        if node.level == 0:
            continue

        if node.is_leaf:
            node.summary = summarizer.summarize_leaf(node.heading, node.content)
            continue

        children_summaries = [
            f"{child.heading}: {child.summary}"
            for child in node.children
            if child.summary
        ]
        if node.content.strip():
            node.summary = summarizer.summarize_parent(
                node.heading,
                children_summaries,
                own_content=node.content,
            )
        else:
            node.summary = summarizer.summarize_parent(node.heading, children_summaries)

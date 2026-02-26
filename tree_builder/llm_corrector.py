"""LLM-assisted level correction for ambiguous heading structures."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol

from tree_builder.preprocessor import RawHeading
from tree_builder.rule_engine import LLM_CONFIDENCE_THRESHOLD, LevelInference
from tree_rag.utils.openai_client import OpenAICompatibleClient


@dataclass(slots=True)
class LLMLevelSuggestion:
    index: int
    level: int
    reasoning: str


class TreeStructureLLMClient(Protocol):
    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        """Return raw text from a chat completion endpoint."""


def build_openai_tree_llm_client_from_env() -> tuple[OpenAICompatibleClient, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for robust tree LLM correction.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    return client, model


def _parse_llm_suggestions(text: str, max_depth: int = 3) -> dict[int, LLMLevelSuggestion]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM correction response is not valid JSON.") from exc

    if isinstance(payload, dict):
        rows = payload.get("results", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise RuntimeError("LLM correction response must be a JSON list/object.")

    suggestions: dict[int, LLMLevelSuggestion] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        level = row.get("level")
        reasoning = row.get("reasoning", "")
        if not isinstance(index, int):
            continue
        if not isinstance(level, int):
            continue
        clamped_level = min(max(level, 1), max_depth)
        suggestions[index] = LLMLevelSuggestion(
            index=index,
            level=clamped_level,
            reasoning=str(reasoning),
        )

    if not suggestions:
        raise RuntimeError("LLM correction returned no usable level suggestions.")

    return suggestions


def _build_full_mode_prompt(
    raw_headings: list[RawHeading],
    rule_results: list[LevelInference],
) -> str:
    heading_lines = "\n".join(
        f"{index}. {heading.raw_text}" for index, heading in enumerate(raw_headings)
    )
    rule_lines = "\n".join(
        f"index={result.signals.index}, level={result.inferred_level}, confidence={result.confidence:.2f}, reason={result.reason}"
        for result in rule_results
    )

    return (
        "You are a document structure analysis expert. Infer the heading level for each heading.\n"
        "Level definitions: 1=top level, 2=subsection, 3=sub-subsection.\n"
        "Output strict JSON object with key 'results'.\n"
        "Each result item must include: index, level, reasoning.\n\n"
        f"Headings:\n{heading_lines}\n\n"
        f"Rule-engine hints:\n{rule_lines}\n"
    )


def _build_partial_mode_prompt(
    raw_headings: list[RawHeading],
    rule_results: list[LevelInference],
    threshold: float,
) -> str:
    context_lines = []
    for result in rule_results:
        marker = "[?]" if result.confidence < threshold else f"[L{result.inferred_level}]"
        context_lines.append(f"{marker} {result.signals.index}. {raw_headings[result.signals.index].raw_text}")

    uncertain_lines = "\n".join(
        f"index={result.signals.index}, heading={raw_headings[result.signals.index].raw_text}"
        for result in rule_results
        if result.confidence < threshold
    )

    context_str = "\n".join(context_lines)
    return (
        "You are a document structure analysis expert.\n"
        "Some headings are uncertain and marked with [?]. Infer only uncertain heading levels.\n"
        "Return strict JSON object with key 'results'.\n"
        "Each item: index, level, reasoning.\n\n"
        f"Structure context:\n{context_str}\n\n"
        f"Uncertain headings:\n{uncertain_lines}\n"
    )


def llm_infer_full_structure(
    raw_headings: list[RawHeading],
    rule_results: list[LevelInference],
    llm_client: TreeStructureLLMClient,
    model: str,
    max_depth: int = 3,
) -> dict[int, LLMLevelSuggestion]:
    prompt = _build_full_mode_prompt(raw_headings, rule_results)
    response = llm_client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _parse_llm_suggestions(response, max_depth=max_depth)


def llm_infer_partial(
    raw_headings: list[RawHeading],
    rule_results: list[LevelInference],
    llm_client: TreeStructureLLMClient,
    model: str,
    threshold: float = LLM_CONFIDENCE_THRESHOLD,
    max_depth: int = 3,
) -> dict[int, LLMLevelSuggestion]:
    prompt = _build_partial_mode_prompt(raw_headings, rule_results, threshold)
    response = llm_client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _parse_llm_suggestions(response, max_depth=max_depth)


def merge_llm_corrections(
    rule_results: list[LevelInference],
    llm_results: dict[int, LLMLevelSuggestion],
    threshold: float = LLM_CONFIDENCE_THRESHOLD,
    keep_rule_conflict_threshold: float = 0.8,
) -> list[LevelInference]:
    merged: list[LevelInference] = []

    for result in rule_results:
        suggestion = llm_results.get(result.signals.index)

        if suggestion is None:
            merged.append(result)
            continue

        if result.confidence >= keep_rule_conflict_threshold and suggestion.level != result.inferred_level:
            merged.append(result)
            continue

        if result.confidence >= threshold:
            merged.append(result)
            continue

        merged.append(
            LevelInference(
                signals=result.signals,
                inferred_level=suggestion.level,
                confidence=0.85,
                reason=f"llm:{suggestion.reasoning}" if suggestion.reasoning else "llm:corrected",
                source="llm",
            )
        )

    return merged

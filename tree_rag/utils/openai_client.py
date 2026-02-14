"""OpenAI-compatible HTTP client for chat, embeddings, and rerank."""

from __future__ import annotations

from http.client import IncompleteRead
import json
import logging
import socket
import time
from urllib import error as urlerror
from urllib.parse import urlsplit
from urllib import request


LOGGER = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def _post_json(self, path: str, payload: dict) -> dict:
        if path.startswith("http://") or path.startswith("https://"):
            endpoint = path
        else:
            endpoint = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
            except urlerror.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                # Retry only for transient server-side errors.
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    wait_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "HTTP %d on %s, retrying in %.1fs (attempt %d/%d).",
                        exc.code,
                        path,
                        wait_seconds,
                        attempt,
                        self.max_retries,
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    last_exception = exc
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {details[:300]}") from exc
            except (urlerror.URLError, TimeoutError, socket.timeout, IncompleteRead, ConnectionResetError, OSError) as exc:
                if attempt < self.max_retries:
                    wait_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "Network error on %s: %s. Retrying in %.1fs (attempt %d/%d).",
                        path,
                        exc,
                        wait_seconds,
                        attempt,
                        self.max_retries,
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    last_exception = exc
                    continue
                raise RuntimeError(f"Request failed: {exc}") from exc

            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                # Retry for possibly truncated payloads on unstable connections.
                if attempt < self.max_retries:
                    wait_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "Invalid JSON response on %s, retrying in %.1fs (attempt %d/%d).",
                        path,
                        wait_seconds,
                        attempt,
                        self.max_retries,
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    last_exception = exc
                    continue
                raise RuntimeError("Response is not valid JSON.") from exc

        if last_exception is not None:
            raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_exception}") from last_exception
        raise RuntimeError("Request failed for an unknown reason.")

    def _is_dashscope_host(self) -> bool:
        host = urlsplit(self.base_url).netloc.lower()
        return host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}

    def _dashscope_origin(self) -> str:
        parsed = urlsplit(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError("Invalid base_url for DashScope.")
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _extract_rerank_scores(data: dict, documents: list[str]) -> list[float] | None:
        scores = [0.0 for _ in documents]

        results = data.get("results")
        if not isinstance(results, list):
            output = data.get("output")
            if isinstance(output, dict):
                results = output.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(scores):
                    continue
                score = item.get("relevance_score", item.get("score", 0.0))
                scores[index] = float(score)
            return scores

        data_list = data.get("data")
        if isinstance(data_list, list):
            # Some providers return scores ordered with the input list.
            for idx, item in enumerate(data_list):
                if idx >= len(scores):
                    break
                if isinstance(item, dict):
                    score = item.get("relevance_score", item.get("score", 0.0))
                else:
                    score = item
                scores[idx] = float(score)
            return scores
        return None

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        data = self._post_json("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError) as exc:
            raise RuntimeError("Invalid chat completion response format.") from exc

        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
            return text.strip()

        return str(content).strip()

    def embeddings(self, model: str, texts: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": texts}
        data = self._post_json("/embeddings", payload)
        try:
            vectors = [list(map(float, item["embedding"])) for item in data["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid embeddings response format.") from exc
        return vectors

    def rerank(self, model: str, query: str, documents: list[str], top_n: int | None = None) -> list[float]:
        if self._is_dashscope_host():
            resolved_top_n = top_n if top_n is not None else len(documents)
            payload = {
                "model": model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": resolved_top_n,
                },
            }
            endpoint = f"{self._dashscope_origin()}/api/v1/services/rerank/text-rerank/text-rerank"
            data = self._post_json(endpoint, payload)
            scores = self._extract_rerank_scores(data, documents)
            if scores is None:
                raise RuntimeError("Invalid rerank response format.")
            return scores

        payload: dict = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        data = self._post_json("/rerank", payload)
        scores = self._extract_rerank_scores(data, documents)
        if scores is None:
            raise RuntimeError("Invalid rerank response format.")
        return scores

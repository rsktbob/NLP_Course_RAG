from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    host: str = "http://localhost:11434"

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        url = self.host.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc.reason}") from exc

    def embed_many(self, model: str, texts: list[str], timeout: int = 300) -> list[list[float]]:
        if not texts:
            return []
        try:
            data = self._post_json("/api/embed", {"model": model, "input": texts}, timeout=timeout)
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                return embeddings
        except OllamaError as exc:
            if "HTTP 404" not in str(exc):
                raise

        embeddings: list[list[float]] = []
        for text in texts:
            data = self._post_json("/api/embeddings", {"model": model, "prompt": text}, timeout=timeout)
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise OllamaError("Ollama did not return an embedding.")
            embeddings.append(embedding)
        return embeddings

    def embed(self, model: str, text: str, timeout: int = 120) -> list[float]:
        return self.embed_many(model, [text], timeout=timeout)[0]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        timeout: int = 300,
    ) -> str:
        data = self._post_json(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama did not return a chat message.")
        return content.strip()

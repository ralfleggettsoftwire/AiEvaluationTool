import json
import time
import uuid
from datetime import UTC, datetime

import httpx

from models import RequestConfig, Result

_DONE_SENTINEL = "[DONE]"


class LLMClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model: str | None = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def _fetch_model(self) -> str:
        try:
            resp = await self._http.get(f"{self._base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return str(data["data"][0]["id"])
        except Exception:
            return "default"

    async def _get_model(self) -> str:
        if self._model is None:
            self._model = await self._fetch_model()
        return self._model

    async def complete(self, request: RequestConfig) -> Result:
        timestamp = datetime.now(tz=UTC)
        model = await self._get_model()
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": f"{uuid.uuid4()}\n{request.prompt}"}],
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        start = time.perf_counter()
        ttft_s: float | None = None
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=body,
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    payload = raw_line[len("data:") :].strip()
                    if payload == _DONE_SENTINEL:
                        break
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed SSE chunk: {payload!r}") from exc
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))

        except Exception as exc:
            total_latency_s = time.perf_counter() - start
            return Result(
                timestamp=timestamp,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_s=0.0,
                total_latency_s=total_latency_s,
                tokens_per_sec=0.0,
                error=str(exc),
            )

        total_latency_s = time.perf_counter() - start
        ttft_final = ttft_s if ttft_s is not None else total_latency_s
        tokens_per_sec = completion_tokens / total_latency_s if total_latency_s > 0 else 0.0

        return Result(
            timestamp=timestamp,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft_final,
            total_latency_s=total_latency_s,
            tokens_per_sec=tokens_per_sec,
            error=None,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

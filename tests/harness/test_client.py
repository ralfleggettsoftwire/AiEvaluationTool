import json
from typing import Any

import httpx
import pytest
import respx

from harness.client import LLMClient
from models import RequestConfig


def _sse_lines(*chunks: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _make_chunk(content: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "choices": [{"delta": {"content": content}}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


BASE_URL = "http://fake-vllm"
REQ = RequestConfig(prompt="hello", max_tokens=16)


@respx.mock
@pytest.mark.asyncio
async def test_successful_response_correct_ttft() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    body = _sse_lines(
        _make_chunk("Hello"),
        _make_chunk(" world", usage={"prompt_tokens": 5, "completion_tokens": 2}),
    )
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        result = await client.complete(REQ)

    assert result.error is None
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.ttft_s >= 0.0
    assert result.total_latency_s >= result.ttft_s
    assert result.tokens_per_sec > 0.0


@respx.mock
@pytest.mark.asyncio
async def test_done_sentinel_ends_stream() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    # Only one real chunk before [DONE]; no usage field — tokens should default to 0
    body = b"data: " + json.dumps(_make_chunk("hi")).encode() + b"\n\ndata: [DONE]\n\n"
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        result = await client.complete(REQ)

    assert result.error is None
    assert result.ttft_s >= 0.0


@respx.mock
@pytest.mark.asyncio
async def test_malformed_sse_chunk_returns_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    body = b"data: not-valid-json\n\n"
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        result = await client.complete(REQ)

    assert result.error is not None
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.ttft_s == 0.0
    assert result.tokens_per_sec == 0.0


@respx.mock
@pytest.mark.asyncio
async def test_connection_error_returns_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    async with LLMClient(BASE_URL) as client:
        result = await client.complete(REQ)

    assert result.error is not None
    assert "refused" in result.error
    assert result.timed_out is False
    assert result.prompt_tokens == 0


@respx.mock
@pytest.mark.asyncio
async def test_timeout_returns_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=httpx.ReadTimeout(""))

    async with LLMClient(BASE_URL, timeout=1.0) as client:
        result = await client.complete(REQ)

    assert result.error == "ReadTimeout"
    assert result.timed_out is True
    assert result.prompt_tokens == 0
    assert result.tokens_per_sec == 0.0


@respx.mock
@pytest.mark.asyncio
async def test_model_name_cached_after_first_call() -> None:
    models_route = respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    body = _sse_lines(_make_chunk("x", usage={"prompt_tokens": 1, "completion_tokens": 1}))
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        await client.complete(REQ)
        await client.complete(REQ)

    assert models_route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_api_key_sent_as_bearer_token() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    body = _sse_lines(_make_chunk("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))
    post_route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL, api_key="secret") as client:
        await client.complete(REQ)

    assert post_route.calls[0].request.headers["authorization"] == "Bearer secret"  # type: ignore[reportUnknownMemberType]


@respx.mock
@pytest.mark.asyncio
async def test_no_auth_header_when_api_key_not_set() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    body = _sse_lines(_make_chunk("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))
    post_route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        await client.complete(REQ)

    assert "authorization" not in post_route.calls[0].request.headers  # type: ignore[reportUnknownMemberType]


@respx.mock
@pytest.mark.asyncio
async def test_model_fetch_failure_uses_default() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("no server"))
    body = _sse_lines(_make_chunk("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))
    post_route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    async with LLMClient(BASE_URL) as client:
        result = await client.complete(REQ)

    assert result.error is None
    sent_body = json.loads(post_route.calls[0].request.content)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert sent_body["model"] == "default"

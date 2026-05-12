import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from harness.runner import Runner
from models import RequestConfig, Result


def _make_result(tokens: int = 10) -> Result:
    return Result(
        timestamp=datetime.now(tz=UTC),
        prompt_tokens=5,
        completion_tokens=tokens,
        ttft_s=0.1,
        total_latency_s=1.0,
        tokens_per_sec=float(tokens),
        error=None,
    )


def _make_requests(n: int) -> list[RequestConfig]:
    return [RequestConfig(prompt=f"prompt {i}", max_tokens=16) for i in range(n)]


@pytest.mark.asyncio
async def test_results_returned_in_input_order() -> None:
    results_to_return = [_make_result(i + 1) for i in range(5)]

    async def _complete(req: RequestConfig) -> Result:
        idx = int(req.prompt.split()[-1])
        await asyncio.sleep(0)
        return results_to_return[idx]

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _complete

    runner = Runner(mock_client, max_concurrency=5)
    results = await runner.run(_make_requests(5))

    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.completion_tokens == i + 1


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency() -> None:
    max_concurrency = 3
    in_flight: list[int] = []
    peak: list[int] = []

    async def _complete(_req: RequestConfig) -> Result:
        in_flight.append(1)
        peak.append(len(in_flight))
        await asyncio.sleep(0.01)
        in_flight.pop()
        return _make_result()

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _complete

    runner = Runner(mock_client, max_concurrency=max_concurrency)
    await runner.run(_make_requests(10))

    assert max(peak) <= max_concurrency


@pytest.mark.asyncio
async def test_exception_from_one_request_does_not_stop_run() -> None:
    call_count = 0

    async def _complete(req: RequestConfig) -> Result:
        nonlocal call_count
        call_count += 1
        if "1" in req.prompt:
            raise RuntimeError("intentional failure")
        return _make_result()

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _complete

    runner = Runner(mock_client, max_concurrency=5)
    results = await runner.run(_make_requests(5))

    assert len(results) == 5
    assert call_count == 5
    failed = [r for r in results if r.error is not None]
    assert len(failed) == 1
    assert "intentional failure" in failed[0].error  # type: ignore[index]


@pytest.mark.asyncio
async def test_all_exceptions_still_returns_all_results() -> None:
    async def _complete(_req: RequestConfig) -> Result:
        raise ValueError("always fails")

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _complete

    runner = Runner(mock_client, max_concurrency=2)
    results = await runner.run(_make_requests(4))

    assert len(results) == 4
    assert all(r.error is not None for r in results)


@pytest.mark.asyncio
async def test_metrics_poller_called_during_run() -> None:
    async def _complete(_req: RequestConfig) -> Result:
        await asyncio.sleep(0.05)
        return _make_result()

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _complete

    mock_poller = AsyncMock()
    mock_poller.poll = AsyncMock(return_value={"vllm:num_requests_running": 1.0})

    with patch("harness.runner._METRICS_INTERVAL_S", 0.01):
        runner = Runner(mock_client, max_concurrency=5, metrics_poller=mock_poller)
        await runner.run(_make_requests(3))

    assert mock_poller.poll.call_count >= 1


@pytest.mark.asyncio
async def test_empty_request_list_returns_empty() -> None:
    mock_client = AsyncMock()
    runner = Runner(mock_client, max_concurrency=2)
    results = await runner.run([])
    assert results == []
    mock_client.complete.assert_not_called()

import asyncio
import contextlib
from datetime import UTC, datetime

from harness.metrics import MetricsPoller
from models import RequestConfig, Result

_METRICS_INTERVAL_S = 5.0


class Runner:
    def __init__(
        self,
        client: "ClientProtocol",
        max_concurrency: int,
        metrics_poller: MetricsPoller | None = None,
    ) -> None:
        self._client = client
        self._sem = asyncio.Semaphore(max_concurrency)
        self._metrics_poller = metrics_poller

    async def run(self, requests: list[RequestConfig]) -> list[Result]:
        results: list[Result | None] = [None] * len(requests)

        async def _execute(index: int, req: RequestConfig) -> None:
            async with self._sem:
                try:
                    results[index] = await self._client.complete(req)
                except Exception as exc:
                    results[index] = Result(
                        timestamp=datetime.now(tz=UTC),
                        prompt_tokens=0,
                        completion_tokens=0,
                        ttft_s=0.0,
                        total_latency_s=0.0,
                        tokens_per_sec=0.0,
                        error=str(exc),
                    )

        tasks = [asyncio.create_task(_execute(i, req)) for i, req in enumerate(requests)]

        if self._metrics_poller is not None:
            poll_task = asyncio.create_task(self._poll_loop())
            await asyncio.gather(*tasks)
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
        else:
            await asyncio.gather(*tasks)

        # Every slot is guaranteed to be filled by _execute; the cast is safe.
        return [r for r in results if r is not None]

    async def _poll_loop(self) -> None:
        assert self._metrics_poller is not None
        while True:
            await self._metrics_poller.poll()
            await asyncio.sleep(_METRICS_INTERVAL_S)


class ClientProtocol:
    """Structural protocol for any object with a complete() coroutine."""

    async def complete(self, request: RequestConfig) -> Result: ...

from datetime import UTC, datetime

import httpx
import numpy as np

from models import GpuSample, GpuStats, SummaryStats


class MetricsPoller:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._url = base_url.rstrip("/") + "/metrics"
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._samples: list[GpuSample] = []
        self._checkpoint_idx: int = 0

    async def poll(self) -> dict[str, float]:
        try:
            resp = await self._http.get(self._url)
            resp.raise_for_status()
            data = _parse_prometheus(resp.text)
        except Exception:
            return {}
        else:
            self._samples.append(
                GpuSample(
                    timestamp=datetime.now(tz=UTC),
                    gpu_cache_usage_perc=data.get("vllm:gpu_cache_usage_perc"),
                    num_requests_running=data.get("vllm:num_requests_running"),
                    num_requests_waiting=data.get("vllm:num_requests_waiting"),
                )
            )
            return data

    def get_samples(self) -> list[GpuSample]:
        return list(self._samples)

    def checkpoint(self) -> list[GpuSample]:
        """Return samples collected since the last checkpoint and advance the cursor."""
        samples = self._samples[self._checkpoint_idx :]
        self._checkpoint_idx = len(self._samples)
        return list(samples)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "MetricsPoller":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


_INTERESTING = frozenset(
    {
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    }
)
_MIN_PARTS = 2


def _parse_prometheus(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # metric_name{labels} value [timestamp]  OR  metric_name value [timestamp]
        parts = stripped.split()
        if len(parts) < _MIN_PARTS:
            continue
        name_with_labels = parts[0]
        # Strip label block if present — name is everything before the first '{'
        brace = name_with_labels.find("{")
        metric_name = name_with_labels[:brace] if brace != -1 else name_with_labels
        if metric_name not in _INTERESTING or metric_name in result:
            continue
        try:
            result[metric_name] = float(parts[1])
        except ValueError:
            continue
    return result


def compute_gpu_stats(samples: list[GpuSample]) -> GpuStats:
    """Aggregate a list of GPU poll samples into per-metric summary statistics."""
    if not samples:
        return GpuStats()

    def _stats(values: list[float]) -> SummaryStats | None:
        if not values:
            return None
        arr = np.array(values, dtype=np.float64)
        return SummaryStats(
            mean=float(np.mean(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    return GpuStats(
        gpu_cache_usage_perc=_stats(
            [s.gpu_cache_usage_perc for s in samples if s.gpu_cache_usage_perc is not None]
        ),
        num_requests_running=_stats(
            [s.num_requests_running for s in samples if s.num_requests_running is not None]
        ),
        num_requests_waiting=_stats(
            [s.num_requests_waiting for s in samples if s.num_requests_waiting is not None]
        ),
    )

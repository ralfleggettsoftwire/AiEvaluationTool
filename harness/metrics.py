import httpx


class MetricsPoller:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._url = base_url.rstrip("/") + "/metrics"
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def poll(self) -> dict[str, float]:
        try:
            resp = await self._http.get(self._url)
            resp.raise_for_status()
            return _parse_prometheus(resp.text)
        except Exception:
            return {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "MetricsPoller":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


_INTERESTING = frozenset({"vllm:gpu_cache_usage_perc", "vllm:num_requests_running"})
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

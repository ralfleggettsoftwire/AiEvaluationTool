import httpx
import pytest
import respx

from harness.metrics import MetricsPoller

BASE_URL = "http://fake-vllm"

PROMETHEUS_PAYLOAD = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model="llama3"} 0.42 1234567890
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model="llama3"} 7.0
"""


@respx.mock
@pytest.mark.asyncio
async def test_happy_path_extracts_expected_values() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=PROMETHEUS_PAYLOAD))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert metrics["vllm:gpu_cache_usage_perc"] == pytest.approx(0.42)
    assert metrics["vllm:num_requests_running"] == pytest.approx(7.0)


@respx.mock
@pytest.mark.asyncio
async def test_missing_metric_omitted_gracefully() -> None:
    payload = "vllm:num_requests_running 3.0\n"
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=payload))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert "vllm:num_requests_running" in metrics
    assert "vllm:gpu_cache_usage_perc" not in metrics


@respx.mock
@pytest.mark.asyncio
async def test_http_error_returns_empty_dict() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(503))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert metrics == {}


@respx.mock
@pytest.mark.asyncio
async def test_connection_error_returns_empty_dict() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(side_effect=httpx.ConnectError("refused"))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert metrics == {}


@respx.mock
@pytest.mark.asyncio
async def test_only_first_occurrence_of_metric_used() -> None:
    payload = "vllm:gpu_cache_usage_perc 0.10\nvllm:gpu_cache_usage_perc 0.99\n"
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=payload))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert metrics["vllm:gpu_cache_usage_perc"] == pytest.approx(0.10)


@respx.mock
@pytest.mark.asyncio
async def test_malformed_value_line_skipped() -> None:
    payload = "vllm:gpu_cache_usage_perc NaN-bad\nvllm:num_requests_running 5.0\n"
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=payload))

    async with MetricsPoller(BASE_URL) as poller:
        metrics = await poller.poll()

    assert "vllm:gpu_cache_usage_perc" not in metrics
    assert metrics["vllm:num_requests_running"] == pytest.approx(5.0)

from datetime import UTC, datetime

import httpx
import pytest
import respx

from harness.metrics import MetricsPoller, compute_gpu_stats
from models import GpuSample

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


# --- Sample accumulation and checkpoint tests ---


@respx.mock
@pytest.mark.asyncio
async def test_successful_poll_appends_sample_with_correct_values() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=PROMETHEUS_PAYLOAD))

    async with MetricsPoller(BASE_URL) as poller:
        await poller.poll()
        samples = poller.get_all_samples()

    assert len(samples) == 1
    assert samples[0].gpu_cache_usage_perc == pytest.approx(0.42)
    assert samples[0].num_requests_running == pytest.approx(7.0)


@respx.mock
@pytest.mark.asyncio
async def test_failed_poll_does_not_append_sample() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(503))

    async with MetricsPoller(BASE_URL) as poller:
        await poller.poll()
        samples = poller.get_all_samples()

    assert samples == []


@respx.mock
@pytest.mark.asyncio
async def test_get_all_samples_returns_full_history_after_checkpoint() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=PROMETHEUS_PAYLOAD))

    async with MetricsPoller(BASE_URL) as poller:
        for _ in range(3):
            await poller.poll()
        poller.checkpoint()
        for _ in range(2):
            await poller.poll()
        samples = poller.get_all_samples()

    assert len(samples) == 5


@respx.mock
@pytest.mark.asyncio
async def test_checkpoint_returns_delta_and_advances_cursor() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=PROMETHEUS_PAYLOAD))

    async with MetricsPoller(BASE_URL) as poller:
        for _ in range(3):
            await poller.poll()
        first = poller.checkpoint()
        for _ in range(2):
            await poller.poll()
        second = poller.checkpoint()

    assert len(first) == 3
    assert len(second) == 2


@respx.mock
@pytest.mark.asyncio
async def test_consecutive_checkpoints_return_non_overlapping_slices() -> None:
    respx.get(f"{BASE_URL}/metrics").mock(return_value=httpx.Response(200, text=PROMETHEUS_PAYLOAD))

    async with MetricsPoller(BASE_URL) as poller:
        for _ in range(4):
            await poller.poll()
        first = poller.checkpoint()
        second = poller.checkpoint()
        for _ in range(2):
            await poller.poll()
        third = poller.checkpoint()

    assert len(first) == 4
    assert len(second) == 0
    assert len(third) == 2


# --- compute_gpu_stats tests ---


def test_compute_gpu_stats_empty_returns_all_none_fields() -> None:
    stats = compute_gpu_stats([])

    assert stats.gpu_cache_usage_perc is None
    assert stats.num_requests_running is None
    assert stats.num_requests_waiting is None


def test_compute_gpu_stats_aggregates_values_correctly() -> None:
    samples = [
        GpuSample(
            timestamp=datetime.now(tz=UTC), gpu_cache_usage_perc=0.1, num_requests_running=2.0
        ),
        GpuSample(
            timestamp=datetime.now(tz=UTC), gpu_cache_usage_perc=0.3, num_requests_running=4.0
        ),
        GpuSample(
            timestamp=datetime.now(tz=UTC), gpu_cache_usage_perc=0.5, num_requests_running=6.0
        ),
    ]
    stats = compute_gpu_stats(samples)

    assert stats.gpu_cache_usage_perc is not None
    assert stats.gpu_cache_usage_perc.mean == pytest.approx(0.3)
    assert stats.num_requests_running is not None
    assert stats.num_requests_running.mean == pytest.approx(4.0)
    assert stats.num_requests_waiting is None


def test_compute_gpu_stats_handles_partial_none_fields() -> None:
    samples = [
        GpuSample(
            timestamp=datetime.now(tz=UTC), gpu_cache_usage_perc=0.5, num_requests_running=None
        ),
        GpuSample(
            timestamp=datetime.now(tz=UTC), gpu_cache_usage_perc=None, num_requests_running=3.0
        ),
    ]
    stats = compute_gpu_stats(samples)

    assert stats.gpu_cache_usage_perc is not None
    assert stats.gpu_cache_usage_perc.mean == pytest.approx(0.5)
    assert stats.num_requests_running is not None
    assert stats.num_requests_running.mean == pytest.approx(3.0)

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from experiments.exp5_soak import Exp5Config, Exp5Soak
from models import RequestConfig, Result


def _make_result() -> Result:
    return Result(
        timestamp=datetime.now(tz=UTC),
        prompt_tokens=5,
        completion_tokens=10,
        ttft_s=0.1,
        total_latency_s=1.0,
        tokens_per_sec=10.0,
        error=None,
    )


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("Soak test prompt", encoding="utf-8")
    return p


def test_build_requests_returns_single_request_config(prompt_file: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        request_timeout_s=30.0,
    )
    exp = Exp5Soak(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == 1
    assert isinstance(requests[0], RequestConfig)


def test_build_requests_uses_prompt_content(prompt_file: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        max_tokens=64,
        request_timeout_s=30.0,
    )
    exp = Exp5Soak(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert requests[0].prompt == "Soak test prompt"
    assert requests[0].max_tokens == 64


@pytest.mark.asyncio
async def test_run_each_user_loops_until_duration_elapsed(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        concurrency=2,
        duration_s=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir, "llama3", "g4dn.xlarge")

    async def _run(
        reqs: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        result = _make_result()
        if on_result is not None:
            on_result(result)
        return [result]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    start = time.monotonic()
    summary = await exp.run(mock_runner)
    elapsed = time.monotonic() - start

    # both users ran at least once each; total >= concurrency
    assert summary.total_requests >= 2
    assert elapsed >= 1.0


@pytest.mark.asyncio
async def test_run_sets_concurrency_on_runner(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        concurrency=7,
        duration_s=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir, "llama3", "g4dn.xlarge")

    async def _run(
        reqs: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        result = _make_result()
        if on_result is not None:
            on_result(result)
        return [result]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    mock_runner.set_max_concurrency.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_run_accumulates_results_from_all_users(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        concurrency=3,
        duration_s=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir, "llama3", "g4dn.xlarge")

    async def _run(
        reqs: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        result = _make_result()
        if on_result is not None:
            on_result(result)
        return [result]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    summary = await exp.run(mock_runner)

    assert summary.total_requests == mock_runner.run.call_count


@pytest.mark.asyncio
async def test_run_writes_config_before_first_request(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        concurrency=1,
        duration_s=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir, "llama3", "g4dn.xlarge")

    config_written_before: list[bool] = []

    async def _run(
        requests: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        config_written_before.append((output_dir / "config.yaml").exists())
        result = _make_result()
        if on_result is not None:
            on_result(result)
        return [result]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    assert config_written_before[0] is True


@pytest.mark.asyncio
async def test_run_streams_results_jsonl_incrementally(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        prompt_file=str(prompt_file),
        concurrency=1,
        duration_s=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir, "llama3", "g4dn.xlarge")

    line_counts: list[int] = []

    async def _run(
        reqs: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        result = _make_result()
        if on_result is not None:
            on_result(result)
        path = output_dir / "results.jsonl"
        if path.exists():
            line_counts.append(len(path.read_text(encoding="utf-8").strip().splitlines()))
        return [result]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    assert len(line_counts) >= 1
    assert line_counts == list(range(1, len(line_counts) + 1))

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from experiments.exp4_concurrency import Exp4Concurrency, Exp4Config
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
    p.write_text("Concurrency test prompt", encoding="utf-8")
    return p


def test_build_requests_returns_all_requests(prompt_file: Path) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5, 10],
        requests_per_level=4,
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 12


def test_build_requests_default_levels(prompt_file: Path) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 6 * 10


def test_build_requests_returns_request_config_objects(prompt_file: Path) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 2],
        requests_per_level=3,
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)


@pytest.mark.asyncio
async def test_run_calls_runner_once_per_concurrency_level(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5, 10],
        requests_per_level=2,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir)

    mock_runner = AsyncMock()
    mock_runner.run.return_value = [_make_result(), _make_result()]

    summary = await exp.run(mock_runner)

    assert mock_runner.run.call_count == 3
    assert summary.total_requests == 6


@pytest.mark.asyncio
async def test_run_sets_semaphore_per_level(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5],
        requests_per_level=1,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir)

    sem_values: list[int] = []

    async def _run(requests: list[RequestConfig]) -> list[Result]:
        sem_values.append(mock_runner._sem._value)
        return [_make_result()]

    mock_runner = AsyncMock()
    mock_runner.run.side_effect = _run
    mock_runner._sem = asyncio.Semaphore(99)

    await exp.run(mock_runner)

    assert sem_values == [1, 5]


@pytest.mark.asyncio
async def test_run_writes_config_before_first_runner_call(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp4Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        concurrency_levels=[1],
        requests_per_level=1,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir)

    config_written_before: list[bool] = []

    async def _run(requests: list[RequestConfig]) -> list[Result]:
        config_written_before.append((output_dir / "config.yaml").exists())
        return [_make_result()]

    mock_runner = AsyncMock()
    mock_runner.run.side_effect = _run

    mock_runner._sem = asyncio.Semaphore(1)

    await exp.run(mock_runner)

    assert config_written_before == [True]

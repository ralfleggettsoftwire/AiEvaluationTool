import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

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


def test_build_requests_returns_requests_per_batch(prompt_file: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        requests_per_batch=20,
    )
    exp = Exp5Soak(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 20


def test_build_requests_default_batch_size(prompt_file: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
    )
    exp = Exp5Soak(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 50


def test_build_requests_uses_prompt_content(prompt_file: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        requests_per_batch=3,
        max_tokens=64,
    )
    exp = Exp5Soak(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    for req in requests:
        assert req.prompt == "Soak test prompt"
        assert req.max_tokens == 64


def test_build_requests_returns_request_config_objects(prompt_file: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        requests_per_batch=2,
    )
    exp = Exp5Soak(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)


@pytest.mark.asyncio
async def test_run_loops_until_duration_elapsed(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        duration_s=1,
        requests_per_batch=2,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir)

    call_count = 0

    async def _run(requests: list[RequestConfig]) -> list[Result]:
        nonlocal call_count
        call_count += 1
        return [_make_result(), _make_result()]

    mock_runner = AsyncMock()
    mock_runner.run.side_effect = _run

    start = time.monotonic()
    summary = await exp.run(mock_runner)
    elapsed = time.monotonic() - start

    assert call_count >= 1
    assert summary.total_requests == call_count * 2
    assert elapsed >= 1.0


@pytest.mark.asyncio
async def test_run_writes_config_before_first_runner_call(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        duration_s=0,
        requests_per_batch=1,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir)

    config_written_before: list[bool] = []

    async def _run(requests: list[RequestConfig]) -> list[Result]:
        config_written_before.append((output_dir / "config.yaml").exists())
        return [_make_result()]

    mock_runner = AsyncMock()
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    if config_written_before:
        assert config_written_before[0] is True

    assert (output_dir / "config.yaml").exists()


@pytest.mark.asyncio
async def test_run_accumulates_results_across_batches(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp5Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        duration_s=1,
        requests_per_batch=3,
    )
    output_dir = tmp_path / "out"
    exp = Exp5Soak(config, output_dir)

    mock_runner = AsyncMock()
    mock_runner.run.return_value = [_make_result(), _make_result(), _make_result()]

    summary = await exp.run(mock_runner)

    assert summary.total_requests == mock_runner.run.call_count * 3

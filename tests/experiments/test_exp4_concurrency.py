from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def test_build_requests_total_is_sum_of_level_times_requests_per_user(prompt_file: Path) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5, 10],
        requests_per_user=4,
        request_timeout_s=30.0,
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    # 1*4 + 5*4 + 10*4 = 64
    assert len(requests) == 64


def test_build_requests_default_config(prompt_file: Path) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        request_timeout_s=30.0,
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    # default levels [1,5,10,25,50,100], requests_per_user=10
    expected = sum(level * 10 for level in [1, 5, 10, 25, 50, 100])
    assert len(requests) == expected


def test_build_requests_returns_request_config_objects(prompt_file: Path) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 2],
        requests_per_user=3,
        request_timeout_s=30.0,
    )
    exp = Exp4Concurrency(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)


@pytest.mark.asyncio
async def test_run_calls_runner_once_per_concurrency_level(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5, 10],
        requests_per_user=2,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir, "llama3", "g4dn.xlarge")

    def _side_effect_1(reqs: list[RequestConfig]) -> list[Result]:
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _side_effect_1

    summary = await exp.run(mock_runner)

    assert mock_runner.run.call_count == 3
    # 1*2 + 5*2 + 10*2 = 32
    assert summary.total_requests == 32


@pytest.mark.asyncio
async def test_run_dispatches_level_times_requests_per_user_per_step(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[2, 4],
        requests_per_user=3,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir, "llama3", "g4dn.xlarge")

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    dispatched: list[int] = []

    async def _run(reqs: list[RequestConfig]) -> list[Result]:
        dispatched.append(len(reqs))
        return [_make_result() for _ in reqs]

    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    assert dispatched == [2 * 3, 4 * 3]


@pytest.mark.asyncio
async def test_run_sets_concurrency_per_level(prompt_file: Path, tmp_path: Path) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[1, 5],
        requests_per_user=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir, "llama3", "g4dn.xlarge")

    def _side_effect_2(reqs: list[RequestConfig]) -> list[Result]:
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _side_effect_2

    await exp.run(mock_runner)

    calls = [c.args[0] for c in mock_runner.set_max_concurrency.call_args_list]
    assert calls == [1, 5]


@pytest.mark.asyncio
async def test_run_writes_config_before_first_runner_call(
    prompt_file: Path, tmp_path: Path
) -> None:
    config = Exp4Config(
        prompt_file=str(prompt_file),
        concurrency_levels=[1],
        requests_per_user=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp4Concurrency(config, output_dir, "llama3", "g4dn.xlarge")

    config_written_before: list[bool] = []

    async def _run(requests: list[RequestConfig]) -> list[Result]:
        config_written_before.append((output_dir / "config.yaml").exists())
        return [_make_result() for _ in requests]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.set_max_concurrency = MagicMock()
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    assert config_written_before == [True]

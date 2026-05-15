import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from pydantic import BaseModel

from experiments.base import BaseExperiment
from models import RequestConfig, Result


def _make_result(
    ttft: float,
    latency: float,
    tps: float,
    error: str | None = None,
    timed_out: bool = False,
) -> Result:
    return Result(
        timestamp=datetime.now(tz=UTC),
        prompt_tokens=10,
        completion_tokens=20,
        ttft_s=ttft,
        total_latency_s=latency,
        tokens_per_sec=tps,
        error=error,
        timed_out=timed_out,
    )


class _EmptyCfg(BaseModel):
    pass


class _SimpleExperiment(BaseExperiment):
    def __init__(self, output_dir: Path, results_to_return: list[Result]) -> None:
        super().__init__(_EmptyCfg(), output_dir, "test-model", "g4dn.xlarge")
        self._preset_results = results_to_return

    def build_requests(self) -> list[RequestConfig]:
        return [RequestConfig(prompt="hello", max_tokens=16) for _ in self._preset_results]


def _make_run_side_effect(
    results: list[Result],
) -> Callable[..., object]:
    async def _run(
        requests: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        for r in results:
            if on_result is not None:
                on_result(r)
        return results

    return _run


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.mark.asyncio
async def test_config_yaml_written_before_runner_run(output_dir: Path) -> None:
    written_before_run: list[bool] = []
    results = [_make_result(0.1, 1.0, 10.0)]

    async def _run(
        requests: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        written_before_run.append((output_dir / "config.yaml").exists())
        for r in results:
            if on_result is not None:
                on_result(r)
        return results

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _run

    exp = _SimpleExperiment(output_dir, results)
    await exp.run(mock_runner)

    assert written_before_run == [True], "config.yaml must exist before runner.run() is called"


@pytest.mark.asyncio
async def test_results_jsonl_has_one_line_per_result(output_dir: Path) -> None:
    results = [_make_result(0.1 * i, float(i), float(10 * i)) for i in range(1, 6)]
    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _make_run_side_effect(results)

    exp = _SimpleExperiment(output_dir, results)
    await exp.run(mock_runner)

    lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(results)
    for line in lines:
        parsed = json.loads(line)
        assert "ttft_s" in parsed


@pytest.mark.asyncio
async def test_results_jsonl_written_incrementally(output_dir: Path) -> None:
    results = [_make_result(0.1 * i, float(i), float(i * 10)) for i in range(1, 4)]
    written_counts: list[int] = []

    async def _run(
        requests: list[RequestConfig],
        on_result: Callable[[Result], None] | None = None,
    ) -> list[Result]:
        for r in results:
            if on_result is not None:
                on_result(r)
            path = output_dir / "results.jsonl"
            if path.exists():
                written_counts.append(len(path.read_text(encoding="utf-8").strip().splitlines()))
        return results

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _run

    exp = _SimpleExperiment(output_dir, results)
    await exp.run(mock_runner)

    assert written_counts == [1, 2, 3]


@pytest.mark.asyncio
async def test_summary_json_correct_percentiles(output_dir: Path) -> None:
    ttft_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = [_make_result(t, t * 2, t * 5) for t in ttft_values]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _make_run_side_effect(results)

    exp = _SimpleExperiment(output_dir, results)
    summary = await exp.run(mock_runner)

    arr = [r.ttft_s for r in results]
    expected_p50 = float(np.percentile(arr, 50))
    expected_p95 = float(np.percentile(arr, 95))

    assert abs(summary.ttft.p50 - expected_p50) < 1e-9
    assert abs(summary.ttft.p95 - expected_p95) < 1e-9

    raw = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert abs(raw["ttft"]["p50"] - expected_p50) < 1e-9


@pytest.mark.asyncio
async def test_error_results_counted_in_summary(output_dir: Path) -> None:
    results = [
        _make_result(0.1, 1.0, 10.0),
        _make_result(0.0, 0.0, 0.0, error="ReadTimeout", timed_out=True),
        _make_result(0.2, 2.0, 5.0),
        _make_result(0.0, 0.0, 0.0, error="OOM"),
    ]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _make_run_side_effect(results)

    exp = _SimpleExperiment(output_dir, results)
    summary = await exp.run(mock_runner)

    assert summary.error_count == 2
    assert summary.timeout_error_count == 1
    assert summary.total_requests == 4


@pytest.mark.asyncio
async def test_summary_json_written_to_disk(output_dir: Path) -> None:
    results = [_make_result(0.1, 1.0, 10.0)]
    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _make_run_side_effect(results)

    exp = _SimpleExperiment(output_dir, results)
    await exp.run(mock_runner)

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["model_name"] == "test-model"
    assert data["hardware"] == "g4dn.xlarge"
    assert "ttft" in data
    assert "total_latency" in data
    assert "tokens_per_sec" in data


@pytest.mark.asyncio
async def test_returned_summary_matches_written_json(output_dir: Path) -> None:
    results = [_make_result(0.1 * i, float(i), float(i * 10)) for i in range(1, 4)]
    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _make_run_side_effect(results)

    exp = _SimpleExperiment(output_dir, results)
    summary = await exp.run(mock_runner)

    raw = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary.ttft.mean - raw["ttft"]["mean"]) < 1e-9


@pytest.mark.asyncio
async def test_finalise_skip_results_file_preserves_existing(output_dir: Path) -> None:
    results = [_make_result(0.1, 1.0, 10.0)]
    exp = _SimpleExperiment(output_dir, results)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentinel = output_dir / "results.jsonl"
    sentinel.write_text("sentinel\n", encoding="utf-8")

    started = datetime.now(tz=UTC)
    exp._finalise(results, started, started, skip_results_file=True)  # type: ignore[reportPrivateUsage]

    assert sentinel.read_text(encoding="utf-8") == "sentinel\n"

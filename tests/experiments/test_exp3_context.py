from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from experiments.exp3_context import Exp3Config, Exp3Context
from models import RequestConfig, Result


def _make_result() -> Result:
    return Result(
        timestamp=datetime.now(tz=UTC),
        prompt_tokens=10,
        completion_tokens=20,
        ttft_s=0.1,
        total_latency_s=1.0,
        tokens_per_sec=10.0,
        error=None,
    )


@pytest.fixture
def prompt_files(tmp_path: Path) -> list[Path]:
    files: list[Path] = []
    for name, content in [
        ("1k.txt", "short prompt"),
        ("4k.txt", "medium prompt"),
        ("16k.txt", "long prompt"),
    ]:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        files.append(p)
    return files


def test_build_requests_repeats_each_file(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=3,
        request_timeout_s=30.0,
    )
    exp = Exp3Context(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == 9


def test_build_requests_total_count(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=5,
        max_tokens=64,
        request_timeout_s=30.0,
    )
    exp = Exp3Context(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == len(prompt_files) * 5


def test_build_requests_prompt_content_matches_files(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=2,
        max_tokens=64,
        request_timeout_s=30.0,
    )
    exp = Exp3Context(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    expected_contents = [p.read_text(encoding="utf-8") for p in prompt_files]
    actual_prompts = [r.prompt for r in requests]

    for i, content in enumerate(expected_contents):
        assert actual_prompts[i * 2] == content
        assert actual_prompts[i * 2 + 1] == content


def test_build_requests_default_repeats(tmp_path: Path) -> None:
    p = tmp_path / "p.txt"
    p.write_text("x", encoding="utf-8")
    config = Exp3Config(
        prompt_files=[str(p)],
        request_timeout_s=30.0,
    )
    exp = Exp3Context(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == 3


def test_build_requests_returns_request_config_objects(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=1,
        request_timeout_s=30.0,
    )
    exp = Exp3Context(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)


# --- run() tests ---


@pytest.mark.asyncio
async def test_run_creates_per_prompt_subdirectories(
    prompt_files: list[Path], tmp_path: Path
) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp3Context(config, output_dir, "llama3", "g4dn.xlarge")

    def _side_effect_1(reqs: list[RequestConfig]) -> list[Result]:
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _side_effect_1

    await exp.run(mock_runner)

    subdirs = sorted(d.name for d in output_dir.iterdir() if d.is_dir())
    assert subdirs == ["00_1k", "01_4k", "02_16k"]


@pytest.mark.asyncio
async def test_run_each_subdir_has_results_and_summary(
    prompt_files: list[Path], tmp_path: Path
) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=2,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp3Context(config, output_dir, "llama3", "g4dn.xlarge")

    def _side_effect_2(reqs: list[RequestConfig]) -> list[Result]:
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _side_effect_2

    await exp.run(mock_runner)

    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            assert (subdir / "results.jsonl").exists(), f"missing results.jsonl in {subdir.name}"
            assert (subdir / "summary.json").exists(), f"missing summary.json in {subdir.name}"


@pytest.mark.asyncio
async def test_run_top_level_summary_aggregates_all_results(
    prompt_files: list[Path], tmp_path: Path
) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=2,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp3Context(config, output_dir, "llama3", "g4dn.xlarge")

    def _side_effect_3(reqs: list[RequestConfig]) -> list[Result]:
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _side_effect_3

    summary = await exp.run(mock_runner)

    assert summary.total_requests == len(prompt_files) * 2


@pytest.mark.asyncio
async def test_run_config_yaml_written_before_first_request(
    prompt_files: list[Path], tmp_path: Path
) -> None:
    config = Exp3Config(
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=1,
        request_timeout_s=30.0,
    )
    output_dir = tmp_path / "out"
    exp = Exp3Context(config, output_dir, "llama3", "g4dn.xlarge")

    config_written: list[bool] = []

    async def _run(reqs: list[RequestConfig]) -> list[Result]:
        config_written.append((output_dir / "config.yaml").exists())
        return [_make_result() for _ in reqs]

    mock_runner = AsyncMock()
    mock_runner.metrics_poller = None
    mock_runner.run.side_effect = _run

    await exp.run(mock_runner)

    assert config_written[0] is True

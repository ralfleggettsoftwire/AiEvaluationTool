from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from harness.local_runner import run_from_config
from models import Result


def _make_result() -> Result:
    return Result(
        timestamp=datetime.now(tz=UTC),
        prompt_tokens=10,
        completion_tokens=20,
        ttft_s=0.1,
        total_latency_s=1.0,
        tokens_per_sec=20.0,
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test prompt", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "experiment_type: exp1_baseline\n"
        "model_name: llama3\n"
        "hardware: g4dn.xlarge\n"
        f"prompt_file: {prompt}\n"
        "n_requests: 1\n"
        "request_timeout_s: 30.0\n",
        encoding="utf-8",
    )
    return config


def _mock_llm_client(result: Result) -> MagicMock:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


async def test_run_from_config_uploads_to_s3_when_bucket_set(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT_URL", "http://localhost:8000")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.chdir(tmp_path)

    with (
        patch("harness.local_runner._probe_metrics", AsyncMock(return_value=False)),
        patch("harness.local_runner.LLMClient", return_value=_mock_llm_client(_make_result())),
        patch("harness.local_runner.S3Manager") as mock_s3_cls,
    ):
        await run_from_config(config_file)

    mock_s3_cls.assert_called_once_with("test-bucket", "eu-west-1")
    mock_s3_cls.return_value.upload_directory.assert_called_once()
    call_args = mock_s3_cls.return_value.upload_directory.call_args
    s3_prefix: str = call_args[0][1]
    assert s3_prefix.startswith("results/llama3/g4dn.xlarge/")
    assert ":" not in s3_prefix, "S3 prefix must not contain ':' (breaks mkdir on macOS)"


async def test_run_from_config_skips_s3_when_no_bucket(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT_URL", "http://localhost:8000")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.chdir(tmp_path)

    with (
        patch("harness.local_runner._probe_metrics", AsyncMock(return_value=False)),
        patch("harness.local_runner.LLMClient", return_value=_mock_llm_client(_make_result())),
        patch("harness.local_runner.S3Manager") as mock_s3_cls,
    ):
        await run_from_config(config_file)

    mock_s3_cls.assert_not_called()

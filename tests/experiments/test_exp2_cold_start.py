from pathlib import Path

import pytest

from experiments.exp2_cold_start import Exp2ColdStart, Exp2Config
from models import RequestConfig


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("Cold start prompt text", encoding="utf-8")
    return p


def test_build_requests_returns_n_warmup_requests(prompt_file: Path) -> None:
    config = Exp2Config(
        prompt_file=str(prompt_file),
        n_warmup_requests=5,
        request_timeout_s=30.0,
    )
    exp = Exp2ColdStart(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == 5


def test_build_requests_uses_prompt_file_content(prompt_file: Path) -> None:
    config = Exp2Config(
        prompt_file=str(prompt_file),
        n_warmup_requests=2,
        max_tokens=32,
        request_timeout_s=30.0,
    )
    exp = Exp2ColdStart(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    for req in requests:
        assert req.prompt == "Cold start prompt text"
        assert req.max_tokens == 32


def test_build_requests_warmup_count_respected(prompt_file: Path) -> None:
    config = Exp2Config(
        prompt_file=str(prompt_file),
        n_warmup_requests=3,
        request_timeout_s=30.0,
    )
    exp = Exp2ColdStart(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert len(requests) == 3


def test_build_requests_returns_request_config_objects(prompt_file: Path) -> None:
    config = Exp2Config(
        prompt_file=str(prompt_file),
        n_warmup_requests=2,
        request_timeout_s=30.0,
    )
    exp = Exp2ColdStart(config, Path("/tmp/unused"), "llama3", "g4dn.xlarge")
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)

from pathlib import Path

import pytest

from experiments.exp1_baseline import Exp1Baseline, Exp1Config
from models import RequestConfig


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("Hello world prompt", encoding="utf-8")
    return p


def test_build_requests_returns_n_requests(prompt_file: Path) -> None:
    config = Exp1Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        n_requests=7,
        max_tokens=128,
    )
    exp = Exp1Baseline(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 7


def test_build_requests_all_use_prompt_file_content(prompt_file: Path) -> None:
    config = Exp1Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        n_requests=3,
        max_tokens=64,
    )
    exp = Exp1Baseline(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    for req in requests:
        assert req.prompt == "Hello world prompt"
        assert req.max_tokens == 64


def test_build_requests_default_n_requests(prompt_file: Path) -> None:
    config = Exp1Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
    )
    exp = Exp1Baseline(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 10


def test_build_requests_returns_request_config_objects(prompt_file: Path) -> None:
    config = Exp1Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_file=str(prompt_file),
        n_requests=2,
    )
    exp = Exp1Baseline(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)

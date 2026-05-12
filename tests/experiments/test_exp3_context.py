from pathlib import Path

import pytest

from experiments.exp3_context import Exp3Config, Exp3Context
from models import RequestConfig


@pytest.fixture
def prompt_files(tmp_path: Path) -> list[Path]:
    files = []
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
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=3,
    )
    exp = Exp3Context(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 9


def test_build_requests_total_count(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=5,
        max_tokens=64,
    )
    exp = Exp3Context(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == len(prompt_files) * 5


def test_build_requests_prompt_content_matches_files(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=2,
        max_tokens=64,
    )
    exp = Exp3Context(config, Path("/tmp/unused"))
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
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files=[str(p)],
    )
    exp = Exp3Context(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 3


def test_build_requests_returns_request_config_objects(prompt_files: list[Path]) -> None:
    config = Exp3Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files=[str(p) for p in prompt_files],
        repeats_per_length=1,
    )
    exp = Exp3Context(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)

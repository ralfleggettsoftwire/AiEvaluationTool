from pathlib import Path

import pytest

from experiments.exp6_workload import Exp6Config, Exp6Workload
from models import RequestConfig


@pytest.fixture
def prompt_files(tmp_path: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for key, content in [
        ("short", "short prompt"),
        ("medium", "medium prompt"),
        ("long", "long prompt"),
    ]:
        p = tmp_path / f"{key}.txt"
        p.write_text(content, encoding="utf-8")
        files[key] = p
    return files


def test_build_requests_returns_n_requests(prompt_files: dict[str, Path]) -> None:
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={k: str(v) for k, v in prompt_files.items()},
        n_requests=50,
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 50


def test_build_requests_default_n_requests(prompt_files: dict[str, Path]) -> None:
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={k: str(v) for k, v in prompt_files.items()},
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 100


def test_build_requests_returns_request_config_objects(prompt_files: dict[str, Path]) -> None:
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={k: str(v) for k, v in prompt_files.items()},
        n_requests=10,
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(isinstance(r, RequestConfig) for r in requests)


def test_build_requests_prompts_come_from_files(prompt_files: dict[str, Path]) -> None:
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={k: str(v) for k, v in prompt_files.items()},
        n_requests=30,
        max_tokens=128,
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    valid_prompts = {"short prompt", "medium prompt", "long prompt"}
    for req in requests:
        assert req.prompt in valid_prompts
        assert req.max_tokens == 128


def test_build_requests_weight_distribution_approximate(prompt_files: dict[str, Path]) -> None:
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={k: str(v) for k, v in prompt_files.items()},
        weights={"short": 1.0, "medium": 0.0, "long": 0.0},
        n_requests=20,
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert all(r.prompt == "short prompt" for r in requests)


def test_build_requests_single_file(tmp_path: Path) -> None:
    p = tmp_path / "only.txt"
    p.write_text("only prompt", encoding="utf-8")
    config = Exp6Config(
        model_name="llama3",
        hardware="g4dn.xlarge",
        prompt_files={"only": str(p)},
        weights={"only": 1.0},
        n_requests=5,
    )
    exp = Exp6Workload(config, Path("/tmp/unused"))
    requests = exp.build_requests()

    assert len(requests) == 5
    assert all(r.prompt == "only prompt" for r in requests)

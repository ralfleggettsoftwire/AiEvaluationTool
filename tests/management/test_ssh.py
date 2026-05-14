from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from management.ssh import SSHManager


@pytest.fixture
def manager() -> SSHManager:
    return SSHManager(host="10.0.0.1", user="ec2-user", key_path="/tmp/key.pem")


def _make_conn() -> MagicMock:
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@patch("management.ssh.Connection")
def test_upload_config_calls_put_with_correct_args(
    mock_conn_cls: MagicMock, manager: SSHManager, tmp_path: Path
) -> None:
    config_file = tmp_path / "exp1.yaml"
    config_file.write_text("model: llama3\n")

    conn = _make_conn()
    mock_conn_cls.return_value = conn

    manager.upload_config(config_file, "/home/ec2-user/config/exp1.yaml")

    conn.put.assert_called_once_with(str(config_file), remote="/home/ec2-user/config/exp1.yaml")


@patch("management.ssh.Connection")
def test_run_experiment_calls_run_with_config_path_and_disown(
    mock_conn_cls: MagicMock, manager: SSHManager
) -> None:
    conn = _make_conn()
    mock_conn_cls.return_value = conn

    manager.run_experiment("/home/ec2-user/config/exp1.yaml")

    conn.run.assert_called_once()
    cmd: str = conn.run.call_args[0][0]
    assert "/home/ec2-user/config/exp1.yaml" in cmd
    assert conn.run.call_args.kwargs.get("disown") is True


@patch("management.ssh.Connection")
def test_run_experiment_quotes_config_path(mock_conn_cls: MagicMock, manager: SSHManager) -> None:
    conn = _make_conn()
    mock_conn_cls.return_value = conn

    manager.run_experiment("/path with spaces/config.yaml")

    cmd: str = conn.run.call_args[0][0]
    assert "'/path with spaces/config.yaml'" in cmd


@patch("management.ssh.Connection")
def test_run_experiment_uses_uv_cli_run_local(
    mock_conn_cls: MagicMock, manager: SSHManager
) -> None:
    conn = _make_conn()
    mock_conn_cls.return_value = conn

    manager.run_experiment("/home/ec2-user/config.yaml")

    cmd: str = conn.run.call_args[0][0]
    assert "cd ~/harness-repo" in cmd
    assert "source ~/.bashrc" in cmd
    assert "uv run python cli.py run-local" in cmd


@patch("management.ssh.Connection")
def test_get_experiment_status_returns_true_when_running(
    mock_conn_cls: MagicMock, manager: SSHManager
) -> None:
    conn = _make_conn()
    mock_conn_cls.return_value = conn
    conn.run.return_value = MagicMock(exited=0)

    assert manager.get_experiment_status() is True


@patch("management.ssh.Connection")
def test_get_experiment_status_returns_false_when_not_running(
    mock_conn_cls: MagicMock, manager: SSHManager
) -> None:
    conn = _make_conn()
    mock_conn_cls.return_value = conn
    conn.run.return_value = MagicMock(exited=1)

    assert manager.get_experiment_status() is False

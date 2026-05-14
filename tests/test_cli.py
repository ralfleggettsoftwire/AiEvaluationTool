from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def ec2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_INSTANCE_ID", "i-1234567890abcdef0")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")


@pytest.fixture
def ssh_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SSH_HOST", "1.2.3.4")
    monkeypatch.setenv("HARNESS_SSH_USER", "ubuntu")
    monkeypatch.setenv("HARNESS_SSH_KEY_PATH", "/home/user/.ssh/id_rsa")


@pytest.fixture
def s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_BUCKET", "my-results-bucket")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")


@pytest.fixture
def temp_config(tmp_path: Path) -> str:
    f = tmp_path / "exp1.yaml"
    f.write_text("experiment: 1\n", encoding="utf-8")
    return str(f)


class TestStart:
    def test_success_prints_ip(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.start.return_value = "54.0.0.1"
        with patch("cli.EC2Manager", mock_manager), patch("cli.find_dotenv", return_value=""):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code == 0
        assert "54.0.0.1" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.start.side_effect = RuntimeError("Instance not found")
        with patch("cli.EC2Manager", mock_manager), patch("cli.find_dotenv", return_value=""):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code != 0

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code == 1
        assert "HARNESS_INSTANCE_ID" in result.output

    def test_updates_dot_env_with_new_ip(
        self, runner: CliRunner, ec2_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("HARNESS_SSH_HOST=old.ip\n", encoding="utf-8")
        mock_manager = MagicMock()
        mock_manager.return_value.start.return_value = "54.0.0.1"
        with (
            patch("cli.EC2Manager", mock_manager),
            patch("cli.find_dotenv", return_value=str(env_file)),
            patch("cli.set_key") as mock_set_key,
        ):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code == 0
        mock_set_key.assert_called_once_with(str(env_file), "HARNESS_SSH_HOST", "54.0.0.1")


class TestStop:
    def test_success_prints_stopped(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code == 0
        assert "stopped" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.stop.side_effect = RuntimeError("API error")
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code == 1
        assert "HARNESS_INSTANCE_ID" in result.output


class TestStatus:
    def test_success_prints_status(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.get_status.return_value = "running"
        mock_manager.return_value.get_public_ip.return_value = "54.0.0.1"
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Status:" in result.output
        assert "running" in result.output
        assert "54.0.0.1" in result.output

    def test_status_with_no_ip(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.get_status.return_value = "stopped"
        mock_manager.return_value.get_public_ip.return_value = None
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Status:" in result.output
        assert "not assigned" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.get_status.side_effect = RuntimeError("API error")
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 1


class TestRun:
    def test_success(self, runner: CliRunner, ssh_env: None, temp_config: str) -> None:
        mock_ssh = MagicMock()
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code == 0
        assert "Experiment started." in result.output

    def test_upload_and_run_called_with_same_remote_path(
        self, runner: CliRunner, ssh_env: None, temp_config: str
    ) -> None:
        mock_ssh_cls = MagicMock()
        mock_instance = mock_ssh_cls.return_value
        with patch("cli.SSHManager", mock_ssh_cls):
            runner.invoke(cli, ["run", "--config", temp_config])
        mock_instance.upload_config.assert_called_once()
        remote_path = mock_instance.upload_config.call_args[0][1]
        mock_instance.run_experiment.assert_called_once_with(remote_path)

    def test_ssh_error_exits_nonzero(
        self, runner: CliRunner, ssh_env: None, temp_config: str
    ) -> None:
        mock_ssh_cls = MagicMock()
        mock_ssh_cls.return_value.upload_config.side_effect = OSError("Connection refused")
        with patch("cli.SSHManager", mock_ssh_cls):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_config_exits_nonzero(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["run", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code != 0

    def test_missing_env_exits_1(self, runner: CliRunner, temp_config: str) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code == 1


class TestDownload:
    def test_no_filters(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 0
        assert "Downloaded to ./results/" in result.output
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/"

    def test_with_model_only(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--model", "llama3"])
        assert result.exit_code == 0
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/llama3/"

    def test_with_model_and_experiment(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--model", "llama3", "--experiment", "1"])
        assert result.exit_code == 0
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/llama3/1/"

    def test_experiment_without_model_exits_1(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--experiment", "1"])
        assert result.exit_code == 1
        assert "--model" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 1
        assert "S3_BUCKET" in result.output


class TestExperimentStatus:
    def test_running_prints_running(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        mock_ssh.return_value.get_experiment_status.return_value = True
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 0
        assert "running" in result.output

    def test_idle_prints_idle(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        mock_ssh.return_value.get_experiment_status.return_value = False
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 0
        assert "idle" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        mock_ssh.return_value.get_experiment_status.side_effect = RuntimeError("SSH error")
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 1

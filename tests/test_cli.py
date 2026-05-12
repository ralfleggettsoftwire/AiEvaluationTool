from __future__ import annotations

import os
import tempfile
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


class TestStart:
    def test_success_prints_ip(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.start.return_value = "54.0.0.1"
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code == 0
        assert "54.0.0.1" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        mock_manager.return_value.start.side_effect = RuntimeError("Instance not found")
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code != 0

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["start"])
        assert result.exit_code == 1
        assert "HARNESS_INSTANCE_ID" in result.output


class TestStop:
    def test_success_prints_stopped(self, runner: CliRunner, ec2_env: None) -> None:
        mock_manager = MagicMock()
        with patch("cli.EC2Manager", mock_manager):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code == 0
        assert "stopped" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["stop"])
        assert result.exit_code == 1


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
        assert "None" in result.output


class TestRun:
    def test_success_with_real_config_file(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        with patch("cli.SSHManager", mock_ssh):
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(b"experiment: 1\n")
                config_path = f.name
            result = runner.invoke(cli, ["run", "--config", config_path])
        assert result.exit_code == 0
        assert "Experiment started." in result.output

    def test_upload_and_run_called_correctly(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh_cls = MagicMock()
        mock_instance = mock_ssh_cls.return_value
        with patch("cli.SSHManager", mock_ssh_cls):
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(b"experiment: 1\n")
                config_path = f.name
            runner.invoke(cli, ["run", "--config", config_path])
        mock_instance.upload_config.assert_called_once()
        mock_instance.run_experiment.assert_called_once_with("~/harness_config.yaml")

    def test_missing_config_exits_nonzero(self, runner: CliRunner, ssh_env: None) -> None:
        mock_ssh = MagicMock()
        with patch("cli.SSHManager", mock_ssh):
            result = runner.invoke(cli, ["run", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code != 0

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(b"experiment: 1\n")
                config_path = f.name
            result = runner.invoke(cli, ["run", "--config", config_path])
        assert result.exit_code == 1


class TestDownload:
    def test_no_filters(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 0
        assert "Downloaded to ./results/" in result.output
        mock_s3.return_value.download_directory.assert_called_once()
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/"

    def test_with_model_only(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--model", "llama3"])
        assert result.exit_code == 0
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/llama3"

    def test_with_model_and_experiment(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--model", "llama3", "--experiment", "1"])
        assert result.exit_code == 0
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/llama3/1"

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 1
        assert "S3_BUCKET" in result.output

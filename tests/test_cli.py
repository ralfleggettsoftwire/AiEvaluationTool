from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

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
    def test_success(self, runner: CliRunner, ec2_env: None, temp_config: str) -> None:
        mock_ssm = MagicMock()
        with patch("cli.SSMManager", mock_ssm):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code == 0
        assert "Experiment started." in result.output

    def test_upload_and_run_called_with_same_remote_path(
        self, runner: CliRunner, ec2_env: None, temp_config: str
    ) -> None:
        mock_ssm_cls = MagicMock()
        mock_instance = mock_ssm_cls.return_value
        with patch("cli.SSMManager", mock_ssm_cls):
            runner.invoke(cli, ["run", "--config", temp_config])
        mock_instance.upload_config.assert_called_once()
        remote_path = mock_instance.upload_config.call_args[0][1]
        mock_instance.run_experiment.assert_called_once_with(remote_path)

    def test_ssm_error_exits_nonzero(
        self, runner: CliRunner, ec2_env: None, temp_config: str
    ) -> None:
        mock_ssm_cls = MagicMock()
        mock_ssm_cls.return_value.upload_config.side_effect = OSError("Connection refused")
        with patch("cli.SSMManager", mock_ssm_cls):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_config_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_ssm = MagicMock()
        with patch("cli.SSMManager", mock_ssm):
            result = runner.invoke(cli, ["run", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code != 0

    def test_missing_env_exits_1(self, runner: CliRunner, temp_config: str) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["run", "--config", temp_config])
        assert result.exit_code == 1
        assert "HARNESS_INSTANCE_ID" in result.output


class TestDownload:
    def test_no_filters(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        mock_s3.return_value.download_directory.return_value = 3
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 0
        assert "3 file(s)" in result.output
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/"

    def test_no_files_found_warns(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        mock_s3.return_value.download_directory.return_value = 0
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download"])
        assert result.exit_code == 0
        assert "no files found" in result.output

    def test_with_model_only(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        mock_s3.return_value.download_directory.return_value = 1
        with patch("cli.S3Manager", mock_s3):
            result = runner.invoke(cli, ["download", "--model", "llama3"])
        assert result.exit_code == 0
        args = mock_s3.return_value.download_directory.call_args[0]
        assert args[0] == "results/llama3/"

    def test_with_model_and_experiment(self, runner: CliRunner, s3_env: None) -> None:
        mock_s3 = MagicMock()
        mock_s3.return_value.download_directory.return_value = 1
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
    def test_running_prints_running(self, runner: CliRunner, ec2_env: None) -> None:
        mock_ssm = MagicMock()
        mock_ssm.return_value.get_experiment_status.return_value = True
        with patch("cli.SSMManager", mock_ssm):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 0
        assert "running" in result.output

    def test_idle_prints_idle(self, runner: CliRunner, ec2_env: None) -> None:
        mock_ssm = MagicMock()
        mock_ssm.return_value.get_experiment_status.return_value = False
        with patch("cli.SSMManager", mock_ssm):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 0
        assert "idle" in result.output

    def test_error_exits_nonzero(self, runner: CliRunner, ec2_env: None) -> None:
        mock_ssm = MagicMock()
        mock_ssm.return_value.get_experiment_status.side_effect = RuntimeError("SSM error")
        with patch("cli.SSMManager", mock_ssm):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_missing_env_exits_1(self, runner: CliRunner) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ["experiment-status"])
        assert result.exit_code == 1
        assert "HARNESS_INSTANCE_ID" in result.output

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from management.ssm import SSMManager

if TYPE_CHECKING:
    from pathlib import Path


def _make_ssm_client() -> MagicMock:
    client = MagicMock()
    client.send_command.return_value = {"Command": {"CommandId": "cmd-test123"}}
    client.get_command_invocation.return_value = {
        "Status": "Success",
        "ResponseCode": 0,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    return client


@patch("management.ssm.boto3.client")
def test_upload_config_sends_base64_write_command(
    mock_boto_client: MagicMock, tmp_path: Path
) -> None:
    ssm_client = _make_ssm_client()
    mock_boto_client.return_value = ssm_client

    config_file = tmp_path / "exp1.yaml"
    config_file.write_text("model: llama3\n", encoding="utf-8")

    mgr = SSMManager("i-1234567890abcdef0")
    mgr.upload_config(config_file, "/home/ec2-user/config.yaml")

    ssm_client.send_command.assert_called_once()
    cmd: str = ssm_client.send_command.call_args[1]["Parameters"]["commands"][0]
    assert "base64 -d" in cmd
    assert "/home/ec2-user/config.yaml" in cmd


@patch("management.ssm.boto3.client")
def test_upload_config_quotes_path_with_spaces(mock_boto_client: MagicMock, tmp_path: Path) -> None:
    ssm_client = _make_ssm_client()
    mock_boto_client.return_value = ssm_client

    config_file = tmp_path / "exp1.yaml"
    config_file.write_text("model: llama3\n", encoding="utf-8")

    mgr = SSMManager("i-1234567890abcdef0")
    mgr.upload_config(config_file, "/home/ec2-user/my config/exp1.yaml")

    cmd: str = ssm_client.send_command.call_args[1]["Parameters"]["commands"][0]
    assert "'/home/ec2-user/my config/exp1.yaml'" in cmd


@patch("management.ssm.boto3.client")
def test_upload_config_raises_on_nonzero_exit(mock_boto_client: MagicMock, tmp_path: Path) -> None:
    ssm_client = _make_ssm_client()
    ssm_client.get_command_invocation.return_value = {
        "Status": "Failed",
        "ResponseCode": 1,
        "StandardOutputContent": "",
        "StandardErrorContent": "Permission denied",
    }
    mock_boto_client.return_value = ssm_client

    config_file = tmp_path / "exp1.yaml"
    config_file.write_text("model: llama3\n", encoding="utf-8")

    mgr = SSMManager("i-1234567890abcdef0")
    with pytest.raises(RuntimeError):
        mgr.upload_config(config_file, "/home/ec2-user/config.yaml")


@patch("management.ssm.boto3.client")
def test_run_experiment_sends_nohup_command(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    mgr.run_experiment("/home/ec2-user/config.yaml")

    ssm_client.send_command.assert_called_once()
    cmd: str = ssm_client.send_command.call_args[1]["Parameters"]["commands"][0]
    assert "nohup" in cmd
    assert "uv run python cli.py run-local" in cmd
    assert "/home/ssm-user/harness-repo" in cmd
    assert "/home/ssm-user/.bashrc" in cmd
    assert "/home/ssm-user/harness.log" in cmd
    assert "~" not in cmd
    ssm_client.get_command_invocation.assert_not_called()


@patch("management.ssm.boto3.client")
def test_run_experiment_quotes_config_path_with_spaces(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    mgr.run_experiment("/home/ec2-user/my config/exp1.yaml")

    cmd: str = ssm_client.send_command.call_args[1]["Parameters"]["commands"][0]
    assert "'/home/ec2-user/my config/exp1.yaml'" in cmd


@patch("management.ssm.boto3.client")
def test_get_experiment_status_returns_true_when_running(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    ssm_client.get_command_invocation.return_value = {
        "Status": "Success",
        "ResponseCode": 0,
        "StandardOutputContent": "12345\n",
        "StandardErrorContent": "",
    }
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    assert mgr.get_experiment_status() is True

    cmd: str = ssm_client.send_command.call_args[1]["Parameters"]["commands"][0]
    assert "cli.py run-local" in cmd


@patch("management.ssm.boto3.client")
def test_get_experiment_status_returns_false_when_not_running(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    ssm_client.get_command_invocation.return_value = {
        "Status": "Failed",
        "ResponseCode": 1,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    assert mgr.get_experiment_status() is False


@patch("management.ssm.boto3.client")
def test_send_and_wait_raises_timeout_error(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    ssm_client.get_command_invocation.return_value = {
        "Status": "InProgress",
        "ResponseCode": -1,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    mgr._timeout = 10.0  # type: ignore[reportPrivateUsage]

    with (
        patch("management.ssm.time.monotonic", side_effect=[0.0, 9999.0]),
        patch("management.ssm.time.sleep"),
        pytest.raises(TimeoutError),
    ):
        mgr._send_and_wait("echo hello")  # type: ignore[reportPrivateUsage]


@patch("management.ssm.boto3.client")
def test_send_and_wait_polls_until_terminal_state(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    in_progress = {
        "Status": "InProgress",
        "ResponseCode": -1,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    success = {
        "Status": "Success",
        "ResponseCode": 0,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    ssm_client.get_command_invocation.side_effect = [in_progress, in_progress, success]
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")

    with patch("management.ssm.time.sleep"):
        rc = mgr._send_and_wait("echo hello")  # type: ignore[reportPrivateUsage]

    assert rc == 0
    assert ssm_client.get_command_invocation.call_count == 3
    assert ssm_client.get_command_invocation.call_args == call(
        CommandId="cmd-test123",
        InstanceId="i-1234567890abcdef0",
    )


@patch("management.ssm.boto3.client")
def test_send_and_wait_cancelling_is_terminal(mock_boto_client: MagicMock) -> None:
    """Status 'Cancelling' is not in _PENDING_STATUSES, so it should be treated as terminal."""
    ssm_client = _make_ssm_client()
    ssm_client.get_command_invocation.return_value = {
        "Status": "Cancelling",
        "ResponseCode": -1,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")

    with patch("management.ssm.time.sleep") as mock_sleep:
        rc = mgr._send_and_wait("echo hello")  # type: ignore[reportPrivateUsage]

    assert rc == -1
    # Must have returned on the first invocation — no sleeping
    mock_sleep.assert_not_called()
    assert ssm_client.get_command_invocation.call_count == 1


@patch("management.ssm.boto3.client")
def test_send_and_wait_retries_on_invocation_does_not_exist(mock_boto_client: MagicMock) -> None:
    ssm_client = _make_ssm_client()
    not_yet = ClientError(
        {"Error": {"Code": "InvocationDoesNotExist", "Message": ""}},
        "GetCommandInvocation",
    )
    success = {
        "Status": "Success",
        "ResponseCode": 0,
        "StandardOutputContent": "",
        "StandardErrorContent": "",
    }
    ssm_client.get_command_invocation.side_effect = [not_yet, not_yet, success]
    mock_boto_client.return_value = ssm_client

    mgr = SSMManager("i-1234567890abcdef0")
    with patch("management.ssm.time.sleep"):
        rc = mgr._send_and_wait("echo hello")  # type: ignore[reportPrivateUsage]

    assert rc == 0
    assert ssm_client.get_command_invocation.call_count == 3


@patch("management.ssm.boto3.client")
def test_upload_config_propagates_client_error(mock_boto_client: MagicMock, tmp_path: Path) -> None:
    """A ClientError from send_command propagates out of upload_config unchanged."""
    ssm_client = _make_ssm_client()
    ssm_client.send_command.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "User is not authorized"}},
        "SendCommand",
    )
    mock_boto_client.return_value = ssm_client

    config_file = tmp_path / "exp1.yaml"
    config_file.write_text("model: llama3\n", encoding="utf-8")

    mgr = SSMManager("i-1234567890abcdef0")
    with pytest.raises(ClientError):
        mgr.upload_config(config_file, "/home/ec2-user/config.yaml")

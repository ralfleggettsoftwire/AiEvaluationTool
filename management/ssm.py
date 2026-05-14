from __future__ import annotations

import base64
import shlex
import time
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from pathlib import Path

    from mypy_boto3_ssm import SSMClient
    from mypy_boto3_ssm.type_defs import GetCommandInvocationResultTypeDef, SendCommandResultTypeDef

_PENDING_STATUSES = {"Pending", "InProgress", "Delayed"}


class SSMManager:
    def __init__(self, instance_id: str, region: str = "eu-west-1") -> None:
        self._instance_id = instance_id
        self._client: SSMClient = boto3.client("ssm", region_name=region)  # type: ignore[reportUnknownMemberType]
        self._poll_interval: float = 2.0
        self._timeout: float = 120.0

    def _send_and_wait(self, command: str) -> int:
        response: SendCommandResultTypeDef = self._client.send_command(
            InstanceIds=[self._instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
        )
        command_id: str = response["Command"]["CommandId"]  # type: ignore[typeddict-item]

        deadline = time.monotonic() + self._timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"SSM command {command_id!r} did not complete within {self._timeout}s"
                )
            invocation: GetCommandInvocationResultTypeDef = self._client.get_command_invocation(
                CommandId=command_id,
                InstanceId=self._instance_id,
            )
            status: str = invocation["Status"]
            if status not in _PENDING_STATUSES:
                return int(invocation["ResponseCode"])
            time.sleep(self._poll_interval)

    def _send_no_wait(self, command: str) -> None:
        self._client.send_command(
            InstanceIds=[self._instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
        )

    def upload_config(self, local_path: Path, remote_path: str) -> None:
        content = local_path.read_text(encoding="utf-8")
        encoded = base64.b64encode(content.encode()).decode()
        quoted_path = shlex.quote(remote_path)
        quoted_encoded = shlex.quote(encoded)
        command = (
            f"mkdir -p $(dirname {quoted_path}) && "
            f"printf '%s' {quoted_encoded} | base64 -d > {quoted_path}"
        )
        rc = self._send_and_wait(command)
        if rc != 0:
            raise RuntimeError(f"Failed to upload config to {remote_path!r} (exit code {rc})")

    def run_experiment(self, config_path: str) -> None:
        quoted = shlex.quote(config_path)
        command = (
            f"cd ~/harness-repo && source ~/.bashrc && "
            f"nohup uv run python cli.py run-local --config {quoted} >> ~/harness.log 2>&1 & disown"
        )
        self._send_no_wait(command)

    def get_experiment_status(self) -> bool:
        rc = self._send_and_wait("pgrep -f 'cli.py run-local'")
        return rc == 0

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from fabric import Connection  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path


class SSHManager:
    def __init__(self, host: str, user: str, key_path: str) -> None:
        self._host = host
        self._user = user
        self._key_path = key_path

    def _connect(self) -> Connection:
        return Connection(
            host=self._host,
            user=self._user,
            connect_kwargs={"key_filename": self._key_path},
        )

    def upload_config(self, local_path: Path, remote_path: str) -> None:
        with self._connect() as conn:
            conn.put(str(local_path), remote=remote_path)

    def run_experiment(self, config_path: str) -> None:
        quoted = shlex.quote(config_path)
        cmd = f"nohup python harness/runner.py --config {quoted} &>/dev/null & disown"
        with self._connect() as conn:
            conn.run(cmd, disown=True)

    def get_experiment_status(self) -> bool:
        with self._connect() as conn:
            result = conn.run("pgrep -f harness/runner.py", hide=True, warn=True)
            return result.exited == 0

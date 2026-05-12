from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import ClientError


class EC2Manager:
    def __init__(self, instance_id: str, region: str = "eu-west-1") -> None:
        self._instance_id = instance_id
        self._client: Any = boto3.client("ec2", region_name=region)

    def _describe(self) -> dict[str, Any]:
        try:
            response: dict[str, Any] = self._client.describe_instances(
                InstanceIds=[self._instance_id]
            )
        except ClientError as exc:
            error_code: str = exc.response["Error"]["Code"]
            if error_code == "InvalidInstanceID.NotFound":
                raise RuntimeError(f"Instance {self._instance_id!r} not found") from exc
            raise
        reservations: list[Any] = response.get("Reservations", [])
        if not reservations:
            raise RuntimeError(f"Instance {self._instance_id!r} not found")
        instance: dict[str, Any] = reservations[0]["Instances"][0]
        return instance

    def start(self) -> str:
        self._describe()
        self._client.start_instances(InstanceIds=[self._instance_id])
        waiter = self._client.get_waiter("instance_running")
        waiter.wait(InstanceIds=[self._instance_id])
        ip: str | None = self.get_public_ip()
        if ip is None:
            raise RuntimeError(f"Instance {self._instance_id!r} is running but has no public IP")
        return ip

    def stop(self) -> None:
        self._describe()
        self._client.stop_instances(InstanceIds=[self._instance_id])
        waiter = self._client.get_waiter("instance_stopped")
        waiter.wait(InstanceIds=[self._instance_id])

    def get_status(self) -> str:
        instance = self._describe()
        return str(instance["State"]["Name"])

    def get_public_ip(self) -> str | None:
        instance = self._describe()
        ip: Any = instance.get("PublicIpAddress")
        return str(ip) if ip is not None else None

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from pathlib import Path

    from mypy_boto3_s3 import S3Client


class S3Manager:
    def __init__(self, bucket: str, region: str = "eu-west-1") -> None:
        self._bucket = bucket
        self._client: S3Client = boto3.client("s3", region_name=region)  # type: ignore[reportUnknownMemberType]

    def upload_directory(self, local_path: Path, s3_prefix: str) -> None:
        files = [p for p in local_path.rglob("*") if p.is_file()]
        if not files:
            raise ValueError(f"No files found under {local_path}")
        for file_path in files:
            relative = file_path.relative_to(local_path)
            key = f"{s3_prefix}/{relative}".replace("\\", "/")
            self._client.upload_file(str(file_path), self._bucket, key)

    def download_directory(self, s3_prefix: str, local_path: Path) -> int:
        keys = self.list_keys(s3_prefix)
        for key in keys:
            relative = key[len(s3_prefix) :].lstrip("/")
            dest = local_path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(dest))
        return len(keys)

    def list_keys(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))  # type: ignore[typeddict-item]
        return keys

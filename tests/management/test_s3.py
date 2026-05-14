from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
import pytest
from moto import mock_aws

from management.s3 import S3Manager

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from mypy_boto3_s3 import S3Client

REGION = "eu-west-1"
BUCKET = "test-results-bucket"


@pytest.fixture
def s3_bucket() -> Generator[S3Client, None, None]:
    with mock_aws():
        client: S3Client = boto3.client("s3", region_name=REGION)  # type: ignore[reportUnknownMemberType]
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        yield client


@pytest.fixture
def manager(s3_bucket: S3Client) -> S3Manager:
    return S3Manager(bucket=BUCKET, region=REGION)


def test_upload_directory_uploads_all_files(tmp_path: Path, manager: S3Manager) -> None:
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world")

    manager.upload_directory(tmp_path, "results/run1")

    keys = manager.list_keys("results/run1")
    assert sorted(keys) == ["results/run1/a.txt", "results/run1/sub/b.txt"]


def test_download_directory_recreates_path_structure(tmp_path: Path, manager: S3Manager) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "file.txt").write_text("content")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "deep.txt").write_text("deep")

    manager.upload_directory(tmp_path / "src", "results/run2")

    dest = tmp_path / "dest"
    manager.download_directory("results/run2", dest)

    assert (dest / "file.txt").read_text() == "content"
    assert (dest / "nested" / "deep.txt").read_text() == "deep"


def test_list_keys_returns_expected_keys(tmp_path: Path, manager: S3Manager) -> None:
    (tmp_path / "x.json").write_text("{}")
    (tmp_path / "y.json").write_text("{}")

    manager.upload_directory(tmp_path, "prefix/run3")

    keys = manager.list_keys("prefix/run3")
    assert sorted(keys) == ["prefix/run3/x.json", "prefix/run3/y.json"]


def test_list_keys_empty_prefix_returns_all(tmp_path: Path, manager: S3Manager) -> None:
    (tmp_path / "one.txt").write_text("1")
    manager.upload_directory(tmp_path, "a")
    manager.upload_directory(tmp_path, "b")

    keys = manager.list_keys()
    assert sorted(keys) == ["a/one.txt", "b/one.txt"]


def test_upload_directory_raises_on_empty_directory(tmp_path: Path, manager: S3Manager) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No files found"):
        manager.upload_directory(empty_dir, "results/run")

import boto3
import pytest
from moto import mock_aws

from management.ec2_manager import EC2Manager

REGION = "eu-west-1"
AMI_ID = "ami-12345678"


def _create_instance(ec2_client) -> str:  # type: ignore[no-untyped-def]
    response = ec2_client.run_instances(ImageId=AMI_ID, MinCount=1, MaxCount=1)
    return response["Instances"][0]["InstanceId"]


@mock_aws
def test_start_returns_ip_and_instance_runs() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    instance_id = _create_instance(ec2)
    ec2.stop_instances(InstanceIds=[instance_id])
    waiter = ec2.get_waiter("instance_stopped")
    waiter.wait(InstanceIds=[instance_id])

    manager = EC2Manager(instance_id, region=REGION)
    ip = manager.start()

    assert isinstance(ip, str)
    assert len(ip) > 0
    assert manager.get_status() == "running"


@mock_aws
def test_stop_waits_until_stopped() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    instance_id = _create_instance(ec2)

    manager = EC2Manager(instance_id, region=REGION)
    manager.stop()

    assert manager.get_status() == "stopped"


@mock_aws
def test_get_status_returns_correct_state() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    instance_id = _create_instance(ec2)

    manager = EC2Manager(instance_id, region=REGION)
    assert manager.get_status() == "running"

    manager.stop()
    assert manager.get_status() == "stopped"


@mock_aws
def test_raises_on_nonexistent_instance_id() -> None:
    manager = EC2Manager("i-nonexistent1234567", region=REGION)

    with pytest.raises(RuntimeError, match="not found"):
        manager.get_status()


@mock_aws
def test_get_public_ip_returns_none_when_stopped() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    instance_id = _create_instance(ec2)

    manager = EC2Manager(instance_id, region=REGION)
    manager.stop()

    assert manager.get_public_ip() is None

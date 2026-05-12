from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from management.ec2_manager import EC2Manager
from management.s3 import S3Manager
from management.ssh import SSHManager


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        click.echo(f"Error: required environment variable {name!r} is not set", err=True)
        sys.exit(1)
    return value


@click.group()
def cli() -> None:
    pass


@cli.command()
def start() -> None:
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    manager = EC2Manager(instance_id, region)
    try:
        ip = manager.start()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(ip)


@cli.command()
def stop() -> None:
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    manager = EC2Manager(instance_id, region)
    manager.stop()
    click.echo("Instance stopped.")


@cli.command()
def status() -> None:
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    manager = EC2Manager(instance_id, region)
    state = manager.get_status()
    ip = manager.get_public_ip()
    click.echo(f"Status: {state}")
    click.echo(f"IP: {ip}")


@cli.command("run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
def run_experiment(config_path: str) -> None:
    host = _require_env("HARNESS_SSH_HOST")
    user = _require_env("HARNESS_SSH_USER")
    key_path = _require_env("HARNESS_SSH_KEY_PATH")
    ssh = SSHManager(host, user, key_path)
    ssh.upload_config(Path(config_path), "~/harness_config.yaml")
    ssh.run_experiment("~/harness_config.yaml")
    click.echo("Experiment started.")


@cli.command()
@click.option("--model", default=None, help="Model name to filter results")
@click.option("--experiment", default=None, help="Experiment number to filter results")
def download(model: str | None, experiment: str | None) -> None:
    bucket = _require_env("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "eu-west-1")

    if model and experiment:
        prefix = f"results/{model}/{experiment}"
    elif model:
        prefix = f"results/{model}"
    else:
        prefix = "results/"

    s3 = S3Manager(bucket, region)
    s3.download_directory(prefix, Path("./results"))
    click.echo("Downloaded to ./results/")


if __name__ == "__main__":
    cli()

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from harness.local_runner import run_from_config
from management.ec2_manager import EC2Manager
from management.s3 import S3Manager
from management.ssm import SSMManager

load_dotenv()

_REMOTE_CONFIG_PATH = "/home/ec2-user/harness_config.yaml"


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
    try:
        manager.stop()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo("Instance stopped.")


@cli.command()
def status() -> None:
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    manager = EC2Manager(instance_id, region)
    try:
        state = manager.get_status()
        ip = manager.get_public_ip()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Status: {state}")
    click.echo(f"IP: {ip if ip is not None else '(not assigned)'}")


@cli.command("run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
def run_experiment(config_path: str) -> None:
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    ssm = SSMManager(instance_id, region)
    try:
        ssm.upload_config(Path(config_path), _REMOTE_CONFIG_PATH)
        ssm.run_experiment(_REMOTE_CONFIG_PATH)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo("Experiment started.")


@cli.command("run-local")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
def run_local(config_path: str) -> None:
    """Run an experiment locally without EC2/SSM infrastructure."""
    asyncio.run(run_from_config(Path(config_path)))


@cli.command()
@click.option("--model", default=None, help="Model name to filter results")
@click.option("--experiment", default=None, help="Experiment number (requires --model)")
def download(model: str | None, experiment: str | None) -> None:
    if experiment and not model:
        click.echo("Error: --experiment requires --model", err=True)
        sys.exit(1)

    bucket = _require_env("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "eu-west-1")

    if model and experiment:
        prefix = f"results/{model}/{experiment}/"
    elif model:
        prefix = f"results/{model}/"
    else:
        prefix = "results/"

    s3 = S3Manager(bucket, region)
    s3.download_directory(prefix, Path("./results"))
    click.echo("Downloaded to ./results/")


@cli.command("experiment-status")
def experiment_status() -> None:
    """Check whether an experiment is currently running on the harness instance."""
    instance_id = _require_env("HARNESS_INSTANCE_ID")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    ssm = SSMManager(instance_id, region)
    try:
        running = ssm.get_experiment_status()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo("running" if running else "idle")


if __name__ == "__main__":
    cli()

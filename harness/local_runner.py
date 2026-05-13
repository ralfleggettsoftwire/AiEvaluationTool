from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models import ExperimentSummary

import httpx
import yaml
from rich.console import Console
from rich.table import Table

from experiments.exp1_baseline import Exp1Baseline, Exp1Config
from experiments.exp2_cold_start import Exp2ColdStart, Exp2Config
from experiments.exp3_context import Exp3Config, Exp3Context
from experiments.exp4_concurrency import Exp4Concurrency, Exp4Config
from experiments.exp5_soak import Exp5Config, Exp5Soak
from experiments.exp6_workload import Exp6Config, Exp6Workload
from harness.client import LLMClient
from harness.metrics import MetricsPoller
from harness.runner import Runner

_console = Console()
_err_console = Console(stderr=False)

_REGISTRY: dict[str, tuple[type[Any], type[Any]]] = {
    "exp1_baseline": (Exp1Config, Exp1Baseline),
    "exp2_cold_start": (Exp2Config, Exp2ColdStart),
    "exp3_context": (Exp3Config, Exp3Context),
    "exp4_concurrency": (Exp4Config, Exp4Concurrency),
    "exp5_soak": (Exp5Config, Exp5Soak),
    "exp6_workload": (Exp6Config, Exp6Workload),
}

_METRICS_PROBE_TIMEOUT = 3.0


async def _probe_metrics(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_METRICS_PROBE_TIMEOUT) as client:
            resp = await client.get(f"{base_url}/metrics")
            return resp.status_code == httpx.codes.OK and "# HELP" in resp.text
    except Exception:
        return False


async def run_from_config(config_path: Path) -> None:
    endpoint = os.environ.get("MODEL_ENDPOINT_URL")
    if not endpoint:
        _err_console.print("[red]Error:[/red] MODEL_ENDPOINT_URL is not set")
        sys.exit(1)
    endpoint = endpoint.rstrip("/")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_type = raw.get("experiment_type")
    if not experiment_type:
        _err_console.print("[red]Error:[/red] config missing 'experiment_type' field")
        sys.exit(1)
    if experiment_type not in _REGISTRY:
        _err_console.print(f"[red]Error:[/red] unknown experiment_type {experiment_type!r}")
        sys.exit(1)

    config_class, experiment_class = _REGISTRY[experiment_type]
    config = config_class.model_validate(raw)

    output_dir = (
        Path("results")
        / config.model_name
        / config.hardware
        / experiment_class.__name__
        / datetime.now(tz=UTC).isoformat()
    )

    _console.print(f"Endpoint  : [cyan]{endpoint}[/cyan]")
    _console.print(f"Experiment: [cyan]{experiment_class.__name__}[/cyan]")
    _console.print(f"Output    : [cyan]{output_dir}[/cyan]")

    has_metrics = await _probe_metrics(endpoint)
    if has_metrics:
        _console.print("Metrics   : [green]available[/green]")
    else:
        _console.print("Metrics   : [yellow]unavailable — GPU stats will not be collected[/yellow]")

    metrics_poller = MetricsPoller(endpoint) if has_metrics else None
    max_concurrency = getattr(config, "concurrency", 1)

    async with LLMClient(endpoint) as client:
        runner = Runner(client, max_concurrency, metrics_poller)
        experiment = experiment_class(config, output_dir)
        summary = await experiment.run(runner)

    _print_summary(summary)


def _print_summary(summary: ExperimentSummary) -> None:
    table = Table(title=f"{summary.experiment} — {summary.model_name} on {summary.hardware}")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    def _fmt(v: float) -> str:
        return f"{v:.3f}"

    table.add_row(
        "TTFT (s)",
        _fmt(summary.ttft.mean),
        _fmt(summary.ttft.p50),
        _fmt(summary.ttft.p95),
        _fmt(summary.ttft.p99),
    )
    table.add_row(
        "Latency (s)",
        _fmt(summary.total_latency.mean),
        _fmt(summary.total_latency.p50),
        _fmt(summary.total_latency.p95),
        _fmt(summary.total_latency.p99),
    )
    table.add_row(
        "Tokens/sec",
        _fmt(summary.tokens_per_sec.mean),
        _fmt(summary.tokens_per_sec.p50),
        _fmt(summary.tokens_per_sec.p95),
        _fmt(summary.tokens_per_sec.p99),
    )

    _console.print(table)
    error_colour = "red" if summary.error_count else "green"
    _console.print(
        f"Requests: {summary.total_requests} total, "
        f"[{error_colour}]{summary.error_count} errors[/{error_colour}]"
    )

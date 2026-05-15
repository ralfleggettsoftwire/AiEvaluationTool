import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from harness.metrics import compute_gpu_stats
from harness.runner import Runner
from models import ExperimentSummary, GpuSample, RequestConfig, Result, SummaryStats


def _compute_stats(values: list[float]) -> SummaryStats:
    if not values:
        return SummaryStats(mean=0.0, p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0)
    arr = np.array(values, dtype=np.float64)
    return SummaryStats(
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


class BaseExperiment(ABC):
    def __init__(self, config: BaseModel, output_dir: Path, model_name: str, hardware: str) -> None:
        self._config = config
        self._output_dir = output_dir
        self._model_name = model_name
        self._hardware = hardware

    @abstractmethod
    def build_requests(self) -> list[RequestConfig]: ...

    def _write_config(self) -> None:
        config_path = self._output_dir / "config.yaml"
        config_path.write_text(
            yaml.dump(self._config.model_dump(mode="json"), default_flow_style=False),
            encoding="utf-8",
        )

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        started_at = datetime.now(tz=UTC)
        results = await runner.run(self.build_requests())
        completed_at = datetime.now(tz=UTC)

        poller = runner.metrics_poller
        return self._finalise(
            results,
            started_at,
            completed_at,
            gpu_samples=poller.get_all_samples() if poller else None,
        )

    def _finalise(
        self,
        results: list[Result],
        started_at: datetime,
        completed_at: datetime,
        output_dir: Path | None = None,
        gpu_samples: list[GpuSample] | None = None,
    ) -> ExperimentSummary:
        resolved_dir = output_dir if output_dir is not None else self._output_dir
        resolved_dir.mkdir(parents=True, exist_ok=True)

        results_path = resolved_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        if gpu_samples:
            metrics_path = resolved_dir / "metrics.jsonl"
            with metrics_path.open("w", encoding="utf-8") as fh:
                for s in gpu_samples:
                    fh.write(s.model_dump_json() + "\n")

        error_count = sum(1 for r in results if r.error is not None)
        timeout_error_count = sum(1 for r in results if r.timed_out)
        successful = [r for r in results if r.error is None]

        summary = ExperimentSummary(
            model_name=self._model_name,
            hardware=self._hardware,
            experiment=type(self).__name__,
            started_at=started_at,
            completed_at=completed_at,
            total_requests=len(results),
            error_count=error_count,
            timeout_error_count=timeout_error_count,
            ttft=_compute_stats([r.ttft_s for r in successful]),
            total_latency=_compute_stats([r.total_latency_s for r in successful]),
            tokens_per_sec=_compute_stats([r.tokens_per_sec for r in successful]),
            gpu_metrics=compute_gpu_stats(gpu_samples) if gpu_samples is not None else None,
        )

        summary_path = resolved_dir / "summary.json"
        summary_path.write_text(
            json.dumps(json.loads(summary.model_dump_json()), indent=2),
            encoding="utf-8",
        )

        return summary

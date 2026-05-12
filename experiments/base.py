import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result, SummaryStats


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
    def __init__(self, config: BaseModel, output_dir: Path) -> None:
        self._config = config
        self._output_dir = output_dir

    @abstractmethod
    def build_requests(self) -> list[RequestConfig]: ...

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._output_dir / "config.yaml"
        config_path.write_text(
            yaml.dump(self._config.model_dump(mode="json"), default_flow_style=False),
            encoding="utf-8",
        )

        started_at = datetime.now(tz=UTC)
        results = await runner.run(self.build_requests())
        completed_at = datetime.now(tz=UTC)

        return self._finalise(results, started_at, completed_at)

    def _finalise(
        self,
        results: list[Result],
        started_at: datetime,
        completed_at: datetime,
    ) -> ExperimentSummary:
        results_path = self._output_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        error_count = sum(1 for r in results if r.error is not None)
        successful = [r for r in results if r.error is None]

        ttft_values = [r.ttft_s for r in successful]
        latency_values = [r.total_latency_s for r in successful]
        tps_values = [r.tokens_per_sec for r in successful]

        config_dict = self._config.model_dump(mode="json")
        summary = ExperimentSummary(
            model_name=str(config_dict["model_name"]),
            hardware=str(config_dict["hardware"]),
            experiment=type(self).__name__,
            started_at=started_at,
            completed_at=completed_at,
            total_requests=len(results),
            error_count=error_count,
            ttft=_compute_stats(ttft_values),
            total_latency=_compute_stats(latency_values),
            tokens_per_sec=_compute_stats(tps_values),
        )

        summary_path = self._output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(json.loads(summary.model_dump_json()), indent=2),
            encoding="utf-8",
        )

        return summary

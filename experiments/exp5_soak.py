import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from experiments.base import BaseExperiment
from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result


class Exp5Config(BaseModel):
    model_name: str
    hardware: str
    prompt_file: str
    max_tokens: int = 128
    concurrency: int = 10
    duration_s: int = 300
    requests_per_batch: int = 50


class Exp5Soak(BaseExperiment):
    def __init__(self, config: Exp5Config, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        return [
            RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
            for _ in range(self._exp_config.requests_per_batch)
        ]

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._output_dir / "config.yaml"
        config_path.write_text(
            yaml.dump(self._exp_config.model_dump(mode="json"), default_flow_style=False),
            encoding="utf-8",
        )

        all_results: list[Result] = []
        started_at = datetime.now(tz=UTC)
        deadline = time.monotonic() + self._exp_config.duration_s

        while time.monotonic() < deadline:
            batch = self.build_requests()
            batch_results = await runner.run(batch)
            all_results.extend(batch_results)

        completed_at = datetime.now(tz=UTC)
        return self._finalise(all_results, started_at, completed_at)

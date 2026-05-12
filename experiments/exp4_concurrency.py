import asyncio
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from experiments.base import BaseExperiment
from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result


class Exp4Config(BaseModel):
    model_name: str
    hardware: str
    prompt_file: str
    max_tokens: int = 128
    concurrency_levels: list[int] = [1, 5, 10, 25, 50, 100]
    requests_per_level: int = 10


class Exp4Concurrency(BaseExperiment):
    def __init__(self, config: Exp4Config, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        total = len(self._exp_config.concurrency_levels) * self._exp_config.requests_per_level
        return [
            RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
            for _ in range(total)
        ]

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._output_dir / "config.yaml"
        config_path.write_text(
            yaml.dump(self._exp_config.model_dump(mode="json"), default_flow_style=False),
            encoding="utf-8",
        )

        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        level_requests = [
            RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
            for _ in range(self._exp_config.requests_per_level)
        ]

        all_results: list[Result] = []
        started_at = datetime.now(tz=UTC)

        for level in self._exp_config.concurrency_levels:
            runner._sem = asyncio.Semaphore(level)
            level_results = await runner.run(level_requests)
            all_results.extend(level_results)

        completed_at = datetime.now(tz=UTC)
        return self._finalise(all_results, started_at, completed_at)

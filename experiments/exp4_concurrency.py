from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result


class Exp4Config(BaseModel):
    prompt_file: str
    max_tokens: int | None = None
    concurrency_levels: list[int] = [1, 5, 10, 25, 50, 100]
    requests_per_user: int = 10
    request_timeout_s: float


class Exp4Concurrency(BaseExperiment):
    def __init__(
        self, config: Exp4Config, output_dir: Path, model_name: str, hardware: str
    ) -> None:
        super().__init__(config, output_dir, model_name, hardware)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        total = sum(
            level * self._exp_config.requests_per_user
            for level in self._exp_config.concurrency_levels
        )
        return [
            RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
            for _ in range(total)
        ]

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        poller = runner.metrics_poller
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        all_results: list[Result] = []
        started_at = datetime.now(tz=UTC)

        for level in self._exp_config.concurrency_levels:
            runner.set_max_concurrency(level)
            level_requests = [
                RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
                for _ in range(level * self._exp_config.requests_per_user)
            ]
            level_started_at = datetime.now(tz=UTC)
            level_results = await runner.run(level_requests)
            level_completed_at = datetime.now(tz=UTC)
            level_samples = poller.checkpoint() if poller else None

            subdir = self._output_dir / f"level_{level}"
            self._finalise(
                level_results,
                level_started_at,
                level_completed_at,
                output_dir=subdir,
                gpu_samples=level_samples,
            )
            all_results.extend(level_results)

        completed_at = datetime.now(tz=UTC)
        return self._finalise(
            all_results,
            started_at,
            completed_at,
            gpu_samples=poller.get_all_samples() if poller else None,
        )

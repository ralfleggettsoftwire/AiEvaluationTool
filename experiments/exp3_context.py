from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result


class Exp3Config(BaseModel):
    model_name: str
    hardware: str
    prompt_files: list[str]
    max_tokens: int | None = None
    repeats_per_length: int = 3
    request_timeout_s: float


class Exp3Context(BaseExperiment):
    def __init__(self, config: Exp3Config, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        requests: list[RequestConfig] = []
        for prompt_file in self._exp_config.prompt_files:
            prompt = Path(prompt_file).read_text(encoding="utf-8")
            requests.extend(
                RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
                for _ in range(self._exp_config.repeats_per_length)
            )
        return requests

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        poller = runner.metrics_poller
        all_results: list[Result] = []
        started_at = datetime.now(tz=UTC)

        for i, prompt_file in enumerate(self._exp_config.prompt_files):
            prompt = Path(prompt_file).read_text(encoding="utf-8")
            level_requests = [
                RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
                for _ in range(self._exp_config.repeats_per_length)
            ]
            level_started_at = datetime.now(tz=UTC)
            level_results = await runner.run(level_requests)
            level_completed_at = datetime.now(tz=UTC)
            level_samples = poller.checkpoint() if poller else None

            subdir = self._output_dir / f"{i:02d}_{Path(prompt_file).stem}"
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

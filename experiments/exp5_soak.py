import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from harness.runner import Runner
from models import ExperimentSummary, RequestConfig, Result


class Exp5Config(BaseModel):
    prompt_file: str
    max_tokens: int | None = None
    concurrency: int = 10
    duration_s: int = 300
    request_timeout_s: float


class Exp5Soak(BaseExperiment):
    def __init__(
        self, config: Exp5Config, output_dir: Path, model_name: str, hardware: str
    ) -> None:
        super().__init__(config, output_dir, model_name, hardware)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        return [RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)]

    async def run(self, runner: Runner) -> ExperimentSummary:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        req = RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)

        all_results: list[Result] = []
        started_at = datetime.now(tz=UTC)
        deadline = time.monotonic() + self._exp_config.duration_s

        runner.set_max_concurrency(self._exp_config.concurrency)

        async def user_loop() -> None:
            while time.monotonic() < deadline:
                results = await runner.run([req])
                all_results.extend(results)

        await asyncio.gather(*[user_loop() for _ in range(self._exp_config.concurrency)])

        completed_at = datetime.now(tz=UTC)
        poller = runner.metrics_poller
        return self._finalise(
            all_results,
            started_at,
            completed_at,
            gpu_samples=poller.get_all_samples() if poller else None,
        )

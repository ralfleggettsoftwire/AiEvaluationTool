from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from models import RequestConfig


class Exp1Config(BaseModel):
    model_name: str
    hardware: str
    prompt_file: str
    max_tokens: int | None = None
    n_requests: int
    request_timeout_s: float


class Exp1Baseline(BaseExperiment):
    def __init__(self, config: Exp1Config, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        prompt = Path(self._exp_config.prompt_file).read_text(encoding="utf-8")
        return [
            RequestConfig(prompt=prompt, max_tokens=self._exp_config.max_tokens)
            for _ in range(self._exp_config.n_requests)
        ]

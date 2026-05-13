import random
from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from models import RequestConfig


class Exp6Config(BaseModel):
    model_name: str
    hardware: str
    prompt_files: dict[str, str]
    weights: dict[str, float]
    max_tokens: int | None = None
    n_requests: int
    concurrency: int


class Exp6Workload(BaseExperiment):
    def __init__(self, config: Exp6Config, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self._exp_config = config

    def build_requests(self) -> list[RequestConfig]:
        keys = list(self._exp_config.prompt_files.keys())
        weight_values = [self._exp_config.weights.get(k, 0.0) for k in keys]
        prompts = {
            k: Path(v).read_text(encoding="utf-8") for k, v in self._exp_config.prompt_files.items()
        }
        chosen_keys = random.choices(keys, weights=weight_values, k=self._exp_config.n_requests)
        return [
            RequestConfig(prompt=prompts[k], max_tokens=self._exp_config.max_tokens)
            for k in chosen_keys
        ]

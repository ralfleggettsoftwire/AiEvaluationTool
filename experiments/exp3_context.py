from pathlib import Path

from pydantic import BaseModel

from experiments.base import BaseExperiment
from models import RequestConfig


class Exp3Config(BaseModel):
    model_name: str
    hardware: str
    prompt_files: list[str]
    max_tokens: int = 128
    repeats_per_length: int = 3


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

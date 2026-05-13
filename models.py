from datetime import datetime

from pydantic import BaseModel


class RequestConfig(BaseModel):
    prompt: str
    max_tokens: int | None = None
    temperature: float = 0.0
    stream: bool = True


class Result(BaseModel):
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_latency_s: float
    tokens_per_sec: float
    error: str | None = None


class SummaryStats(BaseModel):
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


class ExperimentSummary(BaseModel):
    model_name: str
    hardware: str
    experiment: str
    started_at: datetime
    completed_at: datetime
    total_requests: int
    error_count: int
    ttft: SummaryStats
    total_latency: SummaryStats
    tokens_per_sec: SummaryStats

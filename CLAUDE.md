# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A load-test harness for evaluating self-hosted LLMs on AWS EC2 GPU instances. The goal is to find the best **(model, hardware) combinations** for AI-assisted coding across ~400 developers. **Quality/correctness is explicitly out of scope** — this harness is purely a load test with cost accounting. Public benchmarks (HumanEval etc.) cover quality separately.

The model server is **vLLM**, which exposes an OpenAI-compatible REST API (`/v1/chat/completions` with streaming) and a Prometheus `/metrics` endpoint for GPU/VRAM data.

## Commands

This project uses `uv` as the package manager.

```bash
uv run python cli.py <command>     # main entry point (main.py is a placeholder)
uv add <package>                   # add a dependency
uv run pytest                      # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run ruff check .                # lint
uv run ruff format .               # format
uv run pyright .                   # type-check
```

## Code Quality Requirements

All code changes **must** pass the following without errors or warnings:

- **`uv run ruff check .`** — linting (extensive rule set: E/W/F/I/N/UP/B/A/C4/SIM/TCH/ANN/RUF/PT/ERA/PL/TRY/PERF; line length 100)
- **`uv run ruff format .`** — formatting
- **`uv run pyright .`** — strict type checking (pythonVersion=3.12, typeCheckingMode="strict")
- **`uv run pytest`** — all tests must pass, and new behaviour must be covered by tests

## Architecture

```
cli.py                  # click CLI — local entry point; manages EC2/SSH/S3 and local runs
models.py               # all shared Pydantic models (RequestConfig, Result, ExperimentSummary, …)
harness/
  client.py             # LLMClient: async httpx, raw SSE parsing, TTFT measurement
  metrics.py            # MetricsPoller: polls vLLM Prometheus /metrics for GPU/VRAM data
  runner.py             # Runner: asyncio.Semaphore-based concurrency control
  local_runner.py       # run_from_config(): YAML → config class → experiment → results on disk
experiments/
  base.py               # BaseExperiment: abstract; handles result serialisation and aggregation
  exp{1-6}_*.py         # concrete experiments; each has its own Pydantic config class
management/
  ec2_manager.py        # boto3: start/stop harness instance, waiters, public IP
  s3.py                 # boto3: upload/download result directories
  ssh.py                # fabric: config upload, remote experiment trigger
config/                 # example YAML configs (one per experiment type)
prompts/                # static prompt files at 1k/4k/32k/128k token lengths
tests/                  # mirrors source tree; unit tests only, all I/O mocked
```

### Key Patterns

**Config-driven experiment registration.** `local_runner.py` maintains a registry dict mapping `experiment_type` strings to `(ConfigClass, ExperimentClass)` pairs. Adding a new experiment requires registering it there; the CLI and YAML format work automatically.

**Two run modes.** `cli run` uploads a config to the harness EC2 instance and triggers it via SSH (remote mode). `cli run-local` calls `run_from_config()` directly using `MODEL_ENDPOINT_URL` from the environment (local mode, no EC2 needed).

**TTFT precision.** `LLMClient.complete()` records the timestamp of the first `data:` SSE line, not the response open. The `openai` SDK is not used because it may buffer internally.

**GPU metrics are optional.** If the `/metrics` endpoint is unavailable, `MetricsPoller` is `None` and experiments run normally — `summary.json` simply omits `gpu_metrics`.

**Summary stats exclude errors.** `BaseExperiment._finalise()` computes p50/p95/p99 over successful requests only (where `error` is null). Error and timeout counts are reported separately in `summary.json`.

**Experiments with sub-runs write subdirectories.** Exp3 (context length) and Exp4 (concurrency ramp) create a subdirectory per level (e.g. `level_1/`, `00_short_1k/`) and write independent result sets in each.

## Results Directory Structure

```
results/<model_name>/<hardware>/<ExperimentClassName>/<ISO-datetime>/
  config.yaml     # verbatim config written before any requests (partial runs are recoverable)
  results.jsonl   # one JSON line per request: timestamp, prompt_tokens, completion_tokens,
                  #   ttft_s, total_latency_s, tokens_per_sec, error, timed_out
  summary.json    # mean/p50/p95/p99/min/max for successful requests; error + timeout counts
  metrics.jsonl   # GPU samples (only present when /metrics endpoint was available)
```

## Metrics

| Metric | Source |
|--------|--------|
| Tokens/sec | `completion_tokens / total_latency_s` per request; aggregated across requests |
| TTFT | Time from request send to first SSE data chunk, measured in `harness/client.py` |
| Cost per 1k tokens | Instance on-demand hourly rate (AWS Pricing API via boto3) ÷ throughput |
| GPU/VRAM utilisation | vLLM Prometheus endpoint: `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running` |
| Error rate | Count of timeouts, OOM responses, malformed SSE in `results.jsonl` |

## Local CLI Workflow

```bash
uv run python cli.py start                                          # start harness EC2 instance
uv run python cli.py run --config config/exp1_baseline.yaml        # upload config and trigger remotely
uv run python cli.py run-local --config config/exp1_baseline.yaml  # run locally (needs MODEL_ENDPOINT_URL)
uv run python cli.py status                                         # check instance state
uv run python cli.py download [--model llama3 --experiment 1]      # sync results from S3
uv run python cli.py stop                                           # stop harness instance
```

Sensitive values (endpoint URL, AWS credentials, S3 bucket, instance ID) come from environment variables (see `.env.example`), never from config files.

## Testing

Unit tests only — no integration or end-to-end tests. All network I/O is mocked at the library boundary:

- **`respx`** — mocks `httpx` at the transport layer, including chunked SSE responses
- **`moto[ec2,s3]`** — in-process AWS mock for `boto3`
- **`unittest.mock`** — patches `fabric.Connection` for SSH tests

`pytest-asyncio` is configured with `asyncio_mode = "auto"` — all `async def` tests run automatically without `@pytest.mark.asyncio`.

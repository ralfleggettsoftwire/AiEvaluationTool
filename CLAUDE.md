# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A load-test harness for evaluating self-hosted LLMs on AWS EC2 GPU instances. The goal is to find the best **(model, hardware) combinations** for AI-assisted coding across ~400 developers. **Quality/correctness is explicitly out of scope** — this harness is purely a load test with cost accounting. Public benchmarks (HumanEval etc.) cover quality separately.

The model server is **vLLM**, which exposes an OpenAI-compatible REST API (`/v1/chat/completions` with streaming) and a Prometheus `/metrics` endpoint for GPU/VRAM data.

## Commands

This project uses `uv` as the package manager.

```bash
uv run python main.py          # run the application
uv add <package>               # add a dependency
uv run pytest                  # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pyright .               # type-check
```

## Planned Architecture

The codebase is being built from scratch. The planned module structure is:

```
cli.py                  # click CLI — local entry point for managing the EC2 harness instance
management/
  ec2_manager.py        # boto3: start/stop the t3.large harness EC2 instance, waiters
  s3.py                 # boto3: upload results to S3 after each run, download to local machine
  ssh.py                # fabric: SSH into harness instance, upload config, trigger experiments
harness/
  client.py             # httpx async client — raw SSE parsing for precise TTFT measurement
  metrics.py            # poll vLLM Prometheus /metrics endpoint for GPU/VRAM data
  runner.py             # asyncio experiment runner — concurrency control via asyncio.Semaphore
experiments/
  base.py               # base experiment class with shared setup/teardown and result serialisation
  exp1_baseline.py      # single-user baseline
  exp2_cold_start.py    # cold-start timing (starts/stops GPU instance via boto3)
  exp3_context.py       # context length sensitivity (1k–32k tokens)
  exp4_concurrency.py   # concurrency ramp (1–100 concurrent users)
  exp5_soak.py          # sustained load / soak test
  exp6_workload.py      # realistic prompt-type distribution mix
config/                 # YAML experiment config files (one per experiment series)
prompts/                # prompt corpus files at various token lengths
results/                # local mirror of S3 results (gitignored)
```

## Key Design Decisions

**Harness runs on a dedicated EC2 instance (`t3.large`) in the same VPC as the GPU instances.** This removes network variability as a confounding factor when comparing (model, hardware) pairs. After identifying top candidates, a separate short run from representative local developer machines characterises real-world network effects.

**Raw `httpx` instead of the `openai` SDK.** TTFT measurement requires capturing the exact timestamp of the first streamed byte. The `openai` SDK may buffer internally; raw SSE parsing with `httpx` gives unambiguous timing.

**S3 for result persistence.** The harness uploads each run's result directory to S3 immediately after completion. Results survive instance stop/termination and are accessible to the whole team without SSH access to the harness instance.

**Each experiment is independently configured and run.** There is no automated pruning or sequencing between experiments — each is a standalone execution against whatever (model, hardware) combination is currently deployed. Setting up and tearing down the GPU instances themselves is out of scope for this project.

## Results Directory Structure

Results are written on the harness instance and mirrored to S3 with identical paths:

```
results/<model_name>/<hardware>/<experiment_number>/<ISO-datetime>/
  config.yaml     # verbatim copy of the config that produced this run (written first)
  results.jsonl   # one JSON line per request: timestamp, prompt_tokens, completion_tokens,
                  #   ttft_s, total_latency_s, tokens_per_sec, error (if any)
  summary.json    # aggregated stats: mean/p50/p95/p99 for each metric, experiment metadata
```

`config.yaml` is always written before the first request so partial runs are recoverable and always associated with their configuration.

## Metrics

| Metric | Source |
|--------|--------|
| Tokens/sec | `completion_tokens / total_latency_s` per request; aggregated across requests |
| TTFT | Time from request send to first SSE data chunk, measured in `harness/client.py` |
| Cost per 1k tokens | Instance on-demand hourly rate (AWS Pricing API via boto3) ÷ throughput |
| GPU/VRAM utilisation | vLLM Prometheus endpoint: `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running` |
| Error rate | Count of timeouts, OOM responses, malformed SSE in `results.jsonl` |
| Cold-start time | Time from EC2 `StartInstances` call to first successful inference (Experiment 2) |

## Library Inventory

| Library | Role |
|---------|------|
| `httpx` | Async streaming HTTP to model endpoint; raw SSE parsing |
| `boto3` | EC2 lifecycle, S3 upload/download, AWS Pricing API |
| `fabric` | SSH remote control of harness instance (upload config, trigger runs) |
| `click` | Local management CLI |
| `pydantic` | Typed config models and result schemas; each experiment has its own config model |
| `pyyaml` | Experiment config files (one YAML per experiment run) |
| `rich` | Live console output and progress during experiments |
| `pandas` | Post-run aggregation, percentile calculations, pruning comparisons |
| `plotly` | Interactive HTML result charts |

## Local CLI Workflow

```bash
python cli.py start                               # start the stopped harness EC2 instance
python cli.py run --config config/exp1.yaml       # upload config, trigger experiment, stream logs
python cli.py status                              # check if an experiment is running
python cli.py download [--model llama3 --experiment 1]  # sync results from S3 to local ./results/
python cli.py stop                                # stop the harness instance
```

Sensitive values (endpoint URL, AWS credentials, S3 bucket name, instance ID) come from environment variables, never from config files.

## Testing Plan

Unit tests only — no integration or end-to-end tests. All network I/O is mocked at the library boundary.

### Test dependencies

| Library | Role |
|---------|------|
| `pytest` | Test runner |
| `pytest-asyncio` | `async def` test support |
| `respx` | Mock `httpx` at the transport layer (SSE streaming included) |
| `moto` | In-process AWS mock for `boto3` (EC2, S3, Pricing API) |
| `unittest.mock` | Patch `fabric` connections and anything not covered above |

Add with: `uv add --dev pytest pytest-asyncio respx moto`

### Test file layout

Mirror the source tree under `tests/`:

```
tests/
  conftest.py              # shared fixtures: mock httpx transport, moto AWS context, sample configs
  harness/
    test_client.py         # SSE parsing, TTFT measurement, error handling
    test_metrics.py        # Prometheus response parsing
    test_runner.py         # concurrency control, semaphore behaviour, result aggregation
  experiments/
    test_base.py           # config.yaml written before first request; result serialisation
    test_exp1_baseline.py
    test_exp2_cold_start.py
    test_exp3_context.py
    test_exp4_concurrency.py
    test_exp5_soak.py
    test_exp6_workload.py
  management/
    test_ec2_manager.py    # start/stop/waiter logic
    test_s3.py             # upload, download, path construction
    test_ssh.py            # config upload, remote trigger
  test_cli.py              # click CliRunner for all CLI commands
```

### Mocking strategy per module

**`harness/client.py`** — use `respx` to mount a fake SSE transport on the `httpx.AsyncClient`. Return a chunked response that emits `data:` lines on a controlled schedule so TTFT can be asserted precisely without real timing. Test malformed SSE, connection errors, and timeout paths.

**`harness/metrics.py`** — use `respx` to return a static Prometheus text payload. Assert that `gpu_cache_usage_perc` and `num_requests_running` are parsed correctly; test missing/malformed metric lines.

**`harness/runner.py`** — replace `client.complete()` with an `AsyncMock`. Assert that the semaphore limits in-flight requests to the configured concurrency ceiling; assert that all results are collected even when some requests raise.

**`experiments/`** — inject a mock runner (returns pre-canned `Result` objects). Assert that `config.yaml` is written before any request is dispatched. Assert that `results.jsonl` accumulates one line per result and `summary.json` contains correct percentile values.

**`management/ec2_manager.py`** — wrap each test in `@moto.mock_ec2`. Assert start/stop calls reach the correct instance ID; assert that waiters are invoked; assert that the method raises on a non-existent instance ID.

**`management/s3.py`** — wrap each test in `@moto.mock_s3`. Pre-create the bucket in the fixture. Assert that upload writes the exact bytes; assert that download recreates the local path structure; assert that listing returns the expected keys.

**`management/ssh.py`** — patch `fabric.Connection` with `unittest.mock.MagicMock`. Assert that `put()` is called with the correct local and remote paths; assert that `run()` receives the expected command string.

**`cli.py`** — use `click.testing.CliRunner`. Patch the underlying `ec2_manager`, `s3`, and `ssh` callables so no real AWS or SSH calls are made. Assert exit codes and stdout fragments for each subcommand.

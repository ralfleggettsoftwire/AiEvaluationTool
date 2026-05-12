# LLM Evaluation Harness

Load-test harness for evaluating self-hosted LLMs on AWS EC2 GPU instances. The goal is to find the best **(model, hardware) combination** for AI-assisted coding across the team. Quality and correctness of model output are out of scope — those are covered by public benchmarks (HumanEval etc.). This harness is purely a load test with cost accounting.

The model server is **vLLM**, which exposes an OpenAI-compatible REST API and a Prometheus `/metrics` endpoint.

## Architecture

```
Local machine (you)
    │
    │  cli.py  (start / stop / status / run / download)
    ▼
Harness EC2 instance  (t3.large, same VPC as GPU instances)
    │
    │  httpx SSE streaming
    ▼
GPU EC2 instance  (g4dn.xlarge / g5.xlarge / p3.2xlarge etc.)
    └── vLLM serving the model under test
```

The harness instance sits in the same VPC as the GPU instances to eliminate network variability. After identifying top candidates, a short follow-up run from developer machines measures real-world latency.

Results are uploaded to S3 immediately after each run so they survive instance stop/termination and are accessible to the whole team.

## Prerequisites

- Python 3.12+ and [`uv`](https://github.com/astral-sh/uv)
- AWS credentials with EC2, S3, and Pricing API access
- A running vLLM server (the harness does **not** start GPU instances — do that manually)
- A `t3.large` harness EC2 instance already created (pass its ID via `HARNESS_INSTANCE_ID`)

## Setup

```bash
git clone <this-repo>
cd AiEvaluationTools
uv sync
```

## Environment variables

Set these before running any CLI command. A `.env` file is a convenient way to manage them locally (never commit it).

| Variable | Required for | Description |
|----------|-------------|-------------|
| `HARNESS_INSTANCE_ID` | `start`, `stop`, `status` | EC2 instance ID of the harness box (e.g. `i-0abc123`) |
| `HARNESS_SSH_HOST` | `run` | Public IP of the harness instance (available after `start`) |
| `HARNESS_SSH_USER` | `run` | SSH username (typically `ec2-user` or `ubuntu`) |
| `HARNESS_SSH_KEY_PATH` | `run` | Local path to the SSH private key |
| `S3_BUCKET` | `download` | S3 bucket name for result storage |
| `AWS_REGION` | all | AWS region (default: `eu-west-1`) |

## Typical workflow

### 1. Start the harness instance

```bash
python cli.py start
# prints the public IP, e.g. 54.12.34.56
export HARNESS_SSH_HOST=54.12.34.56
```

### 2. Check the harness is reachable

```bash
python cli.py status
# Status: running
# IP: 54.12.34.56
```

### 3. Run an experiment

Pick one of the pre-built configs in `config/`, or write your own (see [Configuring experiments](#configuring-experiments)):

```bash
python cli.py run --config config/exp1_baseline.yaml
# Experiment started.
```

The config is uploaded to the harness instance and the experiment runs in the background. The harness streams logs to disk; use SSH if you need to tail them.

### 4. Download results

```bash
# All results
python cli.py download

# Only results for a specific model
python cli.py download --model llama3-8b

# Only a specific experiment for a model
python cli.py download --model llama3-8b --experiment exp1_baseline
```

Results are written to `./results/` (gitignored) with the same path structure as S3:

```
results/<model_name>/<hardware>/<experiment>/<ISO-datetime>/
  config.yaml     # exact config that produced this run
  results.jsonl   # one JSON line per request
  summary.json    # aggregated stats
```

### 5. Stop the harness instance

```bash
python cli.py stop
```

## Configuring experiments

Each experiment series has its own config schema. The provided configs in `config/` use `llama3-8b` on `g4dn.xlarge` as a starting point — change `model_name` and `hardware` to match what you're testing.

### Experiment 1 — single-user baseline

Sends the same prompt repeatedly from a single user. Use this to get a clean baseline TTFT and tokens/sec before introducing concurrency.

```yaml
# config/exp1_baseline.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/short_1k.txt   # path to a prompt file on the harness instance
max_tokens: 256                      # max tokens to generate per request
n_requests: 20                       # number of sequential requests
```

### Experiment 2 — cold-start timing

Measures warm-up after a cold GPU start. Run this immediately after the GPU instance boots. The cold-start time itself (EC2 `StartInstances` → first successful response) is measured externally; this experiment does the warm-up requests.

```yaml
# config/exp2_cold_start.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/short_1k.txt
max_tokens: 64
n_warmup_requests: 5
```

### Experiment 3 — context length sensitivity

Tests how TTFT and throughput degrade as prompt length grows. Supply one file per context length.

```yaml
# config/exp3_context.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_files:
  - prompts/short_1k.txt    # ~1k tokens
  - prompts/medium_4k.txt   # ~4k tokens
  - prompts/long_16k.txt    # ~16k tokens
  - prompts/xlarge_32k.txt  # ~32k tokens
max_tokens: 128
repeats_per_length: 5       # requests per file
```

### Experiment 4 — concurrency ramp

Ramps concurrent users from 1 to 100, running a fixed number of requests at each level. Shows where throughput saturates and error rates increase.

```yaml
# config/exp4_concurrency.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/short_1k.txt
max_tokens: 128
concurrency_levels: [1, 5, 10, 25, 50, 100]
requests_per_level: 10
```

### Experiment 5 — soak test

Runs at a fixed concurrency for a sustained period. Useful for detecting memory leaks, KV-cache exhaustion, or throughput degradation over time.

```yaml
# config/exp5_soak.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/short_1k.txt
max_tokens: 128
concurrency: 10
duration_s: 300           # run for 5 minutes
requests_per_batch: 50    # requests dispatched per loop iteration
```

### Experiment 6 — realistic workload mix

Samples prompts from multiple files according to weighted probabilities. Approximates the mix of short autocomplete requests, medium chat turns, and long context operations seen in real developer usage.

```yaml
# config/exp6_workload.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_files:
  short: prompts/short_1k.txt
  medium: prompts/medium_4k.txt
  long: prompts/long_16k.txt
weights:
  short: 0.6    # 60% of requests
  medium: 0.3   # 30% of requests
  long: 0.1     # 10% of requests
max_tokens: 256
n_requests: 100
concurrency: 10
```

## Prompt corpus

The `prompts/` directory contains four files at different token lengths, all using a realistic code-review prompt. Approximate sizes:

| File | Tokens |
|------|--------|
| `short_1k.txt` | ~1 000 |
| `medium_4k.txt` | ~4 000 |
| `long_16k.txt` | ~16 000 |
| `xlarge_32k.txt` | ~32 000 |

Replace or extend these with prompts representative of your team's actual usage patterns for more meaningful results.

## Results format

`results.jsonl` — one JSON object per request:

```json
{"timestamp": "2024-01-15T10:23:01Z", "prompt_tokens": 250, "completion_tokens": 180,
 "ttft_s": 0.42, "total_latency_s": 3.1, "tokens_per_sec": 58.1, "error": null}
```

`summary.json` — aggregated stats:

```json
{
  "model_name": "llama3-8b",
  "hardware": "g4dn.xlarge",
  "experiment": "Exp1Baseline",
  "total_requests": 20,
  "error_count": 0,
  "ttft":            {"mean": 0.41, "p50": 0.39, "p95": 0.71, "p99": 0.88, "min": 0.28, "max": 0.91},
  "total_latency":   {"mean": 3.1,  "p50": 3.0,  "p95": 4.8,  "p99": 5.2,  "min": 2.1,  "max": 5.5},
  "tokens_per_sec":  {"mean": 57.0, "p50": 58.0, "p95": 42.0, "p99": 38.0, "min": 36.0, "max": 66.0}
}
```

Stats are computed only over **successful** requests; errored requests are counted separately in `error_count`.

## Testing

```bash
uv run pytest              # all 92 unit tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pyright .           # type-check
```

All tests are unit tests with mocked network I/O — no AWS credentials or running instances required.

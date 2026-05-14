# LLM Evaluation Harness

Load-test harness for evaluating self-hosted LLMs on AWS EC2 GPU instances. The goal is to find the best **(model, hardware) combination** for AI-assisted coding across the team. Quality and correctness of model output are out of scope — those are covered by public benchmarks (HumanEval etc.). This harness is purely a load test with cost accounting.

The model server is **vLLM**, which exposes an OpenAI-compatible REST API and a Prometheus `/metrics` endpoint.

## Architecture

```
Local machine (you)
    │
    │  cli.py  (start / stop / status / run / experiment-status / download)
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

- Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) — on your **local machine**
- AWS credentials with EC2, S3, and Pricing API access — on your **local machine**
- A running vLLM server (the harness does **not** start GPU instances — do that manually)
- A `t3.large` harness EC2 instance **already created** in the same VPC as the GPU instances (its ID goes in `HARNESS_INSTANCE_ID`); see [Harness instance setup](#harness-instance-setup) below
- An S3 bucket **already created** for result storage (its name goes in `S3_BUCKET`)

To create the harness instance and S3 bucket via the AWS CLI (one-time):

```bash
# S3 bucket
aws s3api create-bucket \
  --bucket my-llm-eval-results \
  --region eu-west-1 \
  --create-bucket-configuration LocationConstraint=eu-west-1

# Harness EC2 instance — use an AL2023 AMI; adjust SG and key-pair name
aws ec2 run-instances \
  --image-id <al2023-ami-id-for-your-region> \
  --instance-type t3.large \
  --key-name harness-key \
  --security-group-ids <sg-id> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llm-eval-harness}]'
# Note the InstanceId from the output and put it in .env as HARNESS_INSTANCE_ID
```

## Local setup

Run these commands on your **local machine**:

```bash
git clone <this-repo>
cd AiEvaluationTools
uv sync
cp .env.example .env   # then fill in your values
```

## Harness instance setup

These commands are run **once on the harness EC2 instance** (SSH in after creation). The harness instance is a CPU-only box; it only needs Python, `uv`, and the project code.

```bash
# Install Python 3.12 and uv
sudo dnf install -y python3.12 git
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone the project and install dependencies
git clone <this-repo> ~/harness-repo
cd ~/harness-repo
uv sync

# Persist required environment variables — the experiment runner reads these at runtime
cat >> ~/.bashrc <<'EOF'
export MODEL_ENDPOINT_URL=http://<gpu-instance-private-ip>:8000
export S3_BUCKET=my-llm-eval-results
export AWS_REGION=eu-west-1
EOF
source ~/.bashrc
```

**Networking assumptions this code makes:**
- The harness instance and GPU instance are in the **same VPC** so the harness can reach the model via private IP.
- The GPU instance's security group allows **inbound TCP 8000** from the harness instance's security group.
- The harness instance has an **IAM instance profile** with `s3:PutObject` permission on the results bucket (so experiment results can be uploaded after each run).
- The harness instance is reachable via **SSH on port 22** from your local machine using the key at `HARNESS_SSH_KEY_PATH`.

## Local testing

To verify the harness works before incurring AWS costs, you can run experiments locally on a MacBook using [Ollama](https://ollama.com). See **[local/README.md](local/README.md)** for setup instructions.

## Environment variables

### Local machine (`.env`)

Copy `.env.example` to `.env` and fill in your values. The CLI reads this file on startup. Never commit `.env` — it is gitignored.

`HARNESS_SSH_HOST` is automatically updated in `.env` by `cli.py start` each time the instance gets a new public IP, so you do not need to update it manually.

| Variable | Required for | Description |
|----------|-------------|-------------|
| `HARNESS_INSTANCE_ID` | `start`, `stop`, `status` | EC2 instance ID of the harness box (e.g. `i-0abc123`) |
| `HARNESS_SSH_HOST` | `run`, `experiment-status` | Public IP of the harness instance — auto-updated by `start` |
| `HARNESS_SSH_USER` | `run`, `experiment-status` | SSH username (typically `ec2-user`) |
| `HARNESS_SSH_KEY_PATH` | `run`, `experiment-status` | Local path to the SSH private key |
| `S3_BUCKET` | `download` | S3 bucket name for result storage |
| `AWS_REGION` | all | AWS region (default: `eu-west-1`) |

### Harness instance (`~/.bashrc`)

The experiment runner on the harness instance reads these at runtime (set during [instance setup](#harness-instance-setup)):

| Variable | Description |
|----------|-------------|
| `MODEL_ENDPOINT_URL` | Private IP URL of the vLLM server, e.g. `http://10.0.1.5:8000` |
| `S3_BUCKET` | Same bucket name — results are uploaded here after each run |
| `AWS_REGION` | AWS region (default: `eu-west-1`) |

## Typical workflow

All `cli.py` commands run on your **local machine**. The experiment itself runs on the **harness EC2 instance** in the background, triggered via SSH.

### 1. Start the harness instance

```bash
uv run python cli.py start
# prints the public IP, e.g. 54.12.34.56
# HARNESS_SSH_HOST is automatically updated in .env
```

### 2. Verify the instance is reachable

```bash
uv run python cli.py status
# Status: running
# IP: 54.12.34.56
```

### 3. Run an experiment

Pick one of the pre-built configs in `config/`, or write your own (see [Configuring experiments](#configuring-experiments)):

```bash
uv run python cli.py run --config config/exp1_baseline_small.yaml
# Experiment started.
```

This uploads the config to the harness instance via SSH and starts the experiment in the background. The experiment process on the harness instance logs to `~/harness.log` — SSH in and `tail -f ~/harness.log` if you need to watch progress.

### 4. Monitor until complete

```bash
uv run python cli.py experiment-status
# running   (experiment is still in progress)
# idle      (experiment has finished)
```

Poll this command until it prints `idle`, or tail the log on the harness instance directly:

```bash
# run on the harness instance
tail -f ~/harness.log
```

### 5. Download results

Results are uploaded to S3 automatically when each experiment finishes. Pull them to your local machine:

```bash
# All results
uv run python cli.py download

# Only results for a specific model
uv run python cli.py download --model llama3-8b

# Only a specific experiment for a model
uv run python cli.py download --model llama3-8b --experiment Exp1Baseline
```

Results are written to `./results/` (gitignored) with the same path structure as on S3:

```
results/<model_name>/<hardware>/<ExperimentClassName>/<ISO-datetime>/
  config.yaml     # exact config that produced this run
  results.jsonl   # one JSON line per request
  summary.json    # aggregated stats
```

### 6. Stop the harness instance

```bash
uv run python cli.py stop
```

## Configuring experiments

Each experiment series has its own config schema. The provided configs in `config/` use `llama3-8b` on `g4dn.xlarge` as a starting point — change `model_name` and `hardware` to match what you're testing.

`max_tokens` is optional in most configs. When omitted the model generates until it naturally stops, which gives the most representative throughput signal for a load test. Set it only in two cases: `max_tokens: 1` in Experiment 2 where only TTFT matters; and `max_tokens: 64` in completion-model configs (those ending `_tiny`) where a short cap reflects how production completion endpoints are actually deployed. Omit it everywhere else — capping output prevents the KV cache from filling and gives an unrepresentative throughput reading.

`request_timeout_s` is required in all configs. It sets the per-request read timeout (seconds) passed to the HTTP client. The timeout governs the maximum time the client will wait for the next streamed byte — once streaming begins the timer resets with each chunk, so only the TTFT leg is realistically bounded by this value. 30 seconds is a reasonable threshold for AI-assisted coding tools; requests that exceed it are counted as `timeout_error_count` in the summary and excluded from latency statistics.

### Experiment 1 — single-user baseline

Sends the same prompt repeatedly from a single user. Use this to get a clean baseline TTFT and tokens/sec before introducing concurrency.

```yaml
experiment_type: exp1_baseline
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/small_1k.txt
n_requests: 20
max_tokens: 1024        # optional
request_timeout_s: 30
```

Pre-built configs: `config/exp1_baseline_small.yaml`, `config/exp1_baseline_tiny.yaml`.

### Experiment 2 — cold-start timing

Measures warm-up after a cold GPU start. Run this immediately after the GPU instance boots. The cold-start time itself (EC2 `StartInstances` → first successful response) is measured externally; this experiment does the warm-up requests.

```yaml
# config/exp2_cold_start.yaml
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/small_1k.txt
max_tokens: 1          # Optional, but only TTFT matters here so suggest always setting to 1
n_warmup_requests: 5
request_timeout_s: 30
```

### Experiment 3 — context length sensitivity

Tests how TTFT and throughput degrade as prompt length grows. Supply one file per context length. `max_tokens` is intentionally omitted: each prompt generates its natural response length, which is more representative and avoids artificially truncating the larger prompts (the `medium_4k` task asks for a full rewritten file that comfortably exceeds 1 000 tokens).

```yaml
experiment_type: exp3_context
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_files:
  - prompts/tiny_150.txt
  - prompts/small_1k.txt
  # add further files to extend the curve
max_tokens: 1024        # optional — omit to let each prompt complete naturally
repeats_per_length: 5
request_timeout_s: 30
```

Pre-built config: `config/exp3_context_full.yaml` (all five prompt tiers).

### Experiment 4 — concurrency ramp

Shows where throughput saturates and error rates increase by stepping through a series of concurrency levels in order. At each level the runner's semaphore is set to that value, then `level × requests_per_user` requests are submitted to the asyncio event loop all at once — enough tasks to keep all concurrency slots busy. The semaphore caps how many can be executing simultaneously. The experiment waits for all requests at a level to complete before advancing to the next level, giving each step a clean measurement window. Total requests sent = `sum(level × requests_per_user for level in concurrency_levels)`.

```yaml
experiment_type: exp4_concurrency
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/small_1k.txt
max_tokens: 1024        # optional
concurrency_levels:
  - 1
  - 5
  - 10
  - 25
  # add / remove levels to test at
requests_per_user: 10
request_timeout_s: 30
```

Pre-built configs: `config/exp4_concurrency_small.yaml`, `config/exp4_concurrency_tiny.yaml`.

- **`concurrency_levels`** — ordered list of concurrency values to test. Each entry produces one measurement step.
- **`requests_per_user`** — how many requests each simulated user sends at each step. The total tasks dispatched per step is `level × requests_per_user`, ensuring the concurrency slots are always the binding constraint rather than the task count.

### Experiment 5 — soak test

Useful for detecting memory leaks, KV-cache exhaustion, or throughput degradation over time. The experiment spawns `concurrency` independent user coroutines, each of which issues a request, waits for the response, then immediately issues the next — modelling users who act independently of one another. All coroutines run until the wall-clock deadline (`duration_s` seconds from start) is reached; in-flight requests at the deadline complete normally before the experiment stops.

For the completion-model variant, set `concurrency` to roughly 50% of the saturation level identified in `exp4_concurrency_tiny` — enough to sustain meaningful load without pushing into the error-rate cliff.

```yaml
experiment_type: exp5_soak
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_file: prompts/small_1k.txt
max_tokens: 1024        # optional
concurrency: 10
duration_s: 300
request_timeout_s: 30
```

Pre-built configs: `config/exp5_soak_small.yaml`, `config/exp5_soak_tiny.yaml`.

- **`concurrency`** — number of independent simulated users, each continuously sending requests for the duration of the test.
- **`duration_s`** — wall-clock seconds to sustain load. Each user checks the deadline before issuing its next request, so the experiment stops cleanly without aborting in-flight requests.

### Experiment 6 — realistic workload mix

Approximates a realistic developer request distribution. Before any requests are sent, `n_requests` prompts are sampled from `prompt_files` using `random.choices` weighted by `weights` — producing a randomised but statistically predictable distribution. All requests are then submitted to the asyncio event loop at once, with the `concurrency` semaphore controlling how many are in-flight simultaneously.

```yaml
experiment_type: exp6_workload
model_name: llama3-8b
hardware: g4dn.xlarge
prompt_files:
  small: prompts/small_1k.txt
  medium: prompts/medium_4k.txt
weights:
  small: 0.70
  medium: 0.30
max_tokens: 1024        # optional
n_requests: 100
concurrency: 10
request_timeout_s: 30
```

Pre-built configs: `config/exp6_workload_chat.yaml`, `config/exp6_workload_completion.yaml`, `config/exp6_workload_unified.yaml`.

- **`n_requests`** — total number of requests to send across the entire run. Prompt type for each request is chosen independently by weighted random sampling, so the actual mix converges to the configured proportions as `n_requests` grows.
- **`concurrency`** — maximum number of requests in-flight at the same time.

## Running order

When evaluating a new model–hardware combination, follow one of the tracks below. After the concurrency ramp, check whether the saturation point meets your target — if not, stop there rather than spending time on the soak and workload mix.

### Chat / coding model

| Step | Config | Purpose |
|------|--------|---------|
| 1 | `exp2_cold_start` | Confirms the server responds; sets first-request TTFT expectation |
| 2 | `exp1_baseline_small` | Steady-state throughput and latency reference |
| 3 | `exp3_context_full` | Context scaling curve — reveals collapse at medium/large inputs before committing to longer runs |
| 4 | `exp4_concurrency_small` | **Decision gate** — saturation point for coding tasks; stop here if below target |
| 5 | `exp6_workload_chat` | End-to-end realistic characterisation |
| 6 | `exp5_soak_small` | Stability over 5 min; most time-consuming, run last |

### Completion model

| Step | Config | Purpose |
|------|--------|---------|
| 1 | `exp2_cold_start` | Same sanity check |
| 2 | `exp1_baseline_tiny` | Autocomplete TTFT and throughput reference |
| 3 | `exp4_concurrency_tiny` | **Decision gate** — autocomplete saturation; note the level for step 5 |
| 4 | `exp6_workload_completion` | Realistic characterisation of the autocomplete workload |
| 5 | `exp5_soak_tiny` | Stability; set `concurrency` to ~50% of the saturation level from step 3 |

Experiment 3 is omitted: completion models receive short, fixed-size inputs by design, so a context scaling curve is not meaningful.

### Unified model (single deployment for all task types)

Run the chat track, inserting `exp1_baseline_tiny` after step 2 and `exp4_concurrency_tiny` after step 4. Replace `exp6_workload_chat` with `exp6_workload_unified`.

## Prompt corpus

The `prompts/` directory contains five files at different token lengths, each using a distinct task type representative of real developer usage:

| File | Approx tokens | Task |
|------|--------------|------|
| `tiny_150.txt` | ~150 | Inline line completion (single expression, IDE-style cursor marker) |
| `small_1k.txt` | ~1 000 | Implement a short function from a detailed spec (sliding-window rate limiter) |
| `medium_4k.txt` | ~4 000 | Fix a bug and extend an existing class (async connection pool) |
| `large_32k.txt` | ~32 000 | Diagnose a memory leak given real CPython asyncio source as context |
| `xlarge_128k.txt` | ~128 000 | Implement a feature (async reverse proxy) given 7 stdlib source files as reference |

Different task types produce different output lengths, which gives more realistic throughput signals across the token tiers. Replace or extend these files with prompts representative of your team's actual usage patterns for more meaningful results.

Note that a UUID is prepended to each prompt before it is sent to the model. This prevents prompt caching from generating unrealistic results.

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
  "error_count": 2,
  "timeout_error_count": 1,
  "ttft":            {"mean": 0.41, "p50": 0.39, "p95": 0.71, "p99": 0.88, "min": 0.28, "max": 0.91},
  "total_latency":   {"mean": 3.1,  "p50": 3.0,  "p95": 4.8,  "p99": 5.2,  "min": 2.1,  "max": 5.5},
  "tokens_per_sec":  {"mean": 57.0, "p50": 58.0, "p95": 42.0, "p99": 38.0, "min": 36.0, "max": 66.0}
}
```

`ttft`, `total_latency`, and `tokens_per_sec` statistics are computed **only over successful requests** (those where `error` is null in `results.jsonl`). Failed requests contribute to `error_count` and, if they timed out, to `timeout_error_count`, but are excluded from all latency and throughput percentiles.

## Testing

Run unit tests
```bash
uv run pytest
```
Lint
```bash
uv run ruff check .
```
Format
```bash
uv run ruff format .
```
Type-check
```bash
uv run pyright .
```

All tests are unit tests with mocked network I/O — no AWS credentials or running instances required.

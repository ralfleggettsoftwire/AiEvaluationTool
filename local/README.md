# Local Ollama Setup

Shell script to start a local [Ollama](https://ollama.com) server with `qwen2.5-coder:7b`
for testing the harness on a MacBook before deploying to AWS.

## Prerequisites

- macOS with Apple Silicon (M1 or later)
- [Ollama](https://ollama.com/download) installed — or via Homebrew:
  ```bash
  brew install ollama
  ```

## Setup

**1. Start the server and pull the model:**
```bash
bash ./start_ollama.sh
```
The first run downloads ~4.5 GB of model weights; subsequent runs reuse the local cache.

**2. Set `MODEL_ENDPOINT_URL` in your `.env`:**
```
MODEL_ENDPOINT_URL=http://localhost:11434
```

**3. Run any experiment:**
```bash
cd .. && python cli.py run-local --config config/exp1_baseline.yaml
```

## Stopping

Ollama runs as a background process. To stop it:
```bash
pkill ollama
```

## Known limitations

- **Token counts will be zero.** Ollama's OpenAI-compatible API does not support
  `stream_options: {include_usage: true}`, so `prompt_tokens`, `completion_tokens`,
  and `tokens_per_sec` will be 0 in results. TTFT and total latency are unaffected.
- **No GPU metrics.** Ollama does not expose a Prometheus `/metrics` endpoint.
  The harness will print "Metrics: unavailable" at startup and continue normally.

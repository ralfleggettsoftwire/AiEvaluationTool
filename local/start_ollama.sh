#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen2.5-coder:7b"
OLLAMA_URL="http://localhost:11434"

if ! command -v ollama &>/dev/null; then
    echo "Error: ollama is not installed."
    echo "Install it from https://ollama.com/download or via: brew install ollama"
    exit 1
fi

if ! curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
    echo "Starting ollama server..."
    ollama serve &>/dev/null &
    for i in $(seq 1 30); do
        if curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
            break
        fi
        sleep 1
    done
    if ! curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
        echo "Error: ollama server did not start within 30 seconds"
        exit 1
    fi
    echo "Server started."
else
    echo "Ollama server is already running."
fi

echo "Pulling ${MODEL}..."
ollama pull "${MODEL}"

echo "Warming up ${MODEL}..."
curl -sf "${OLLAMA_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1, \"stream\": false}" \
    > /dev/null
echo "Model loaded and ready."

echo ""
echo "Ready. Set MODEL_ENDPOINT_URL=${OLLAMA_URL} in your .env and run:"
echo "  python cli.py run-local --config config/exp1_baseline.yaml"

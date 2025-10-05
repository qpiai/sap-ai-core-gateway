#!/bin/bash

set -e

echo "Starting SAP AI Core OpenAI Proxy..."

# Check if key.json exists
if [ ! -f "key.json" ]; then
    echo "Error: key.json not found. Please mount it as a volume:"
    echo "  docker run -v /path/to/key.json:/app/key.json ..."
    exit 1
fi

# Export environment variables for child processes
export SAP_PROXY_API_KEY="${SAP_PROXY_API_KEY}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}"

# Start SAP OAuth Proxy in background
echo "Starting SAP OAuth Proxy on port 4000..."
uv run python src/sap_openai_proxy.py &
SAP_PID=$!

# Wait for SAP proxy to be ready
sleep 3

# Start LiteLLM Proxy in foreground
echo "Starting LiteLLM Proxy on port 8082..."
exec uv run litellm --config config/litellm_config.yaml --port 8082

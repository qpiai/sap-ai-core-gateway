#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for --with-litellm flag
WITH_LITELLM=false
if [[ "$1" == "--with-litellm" ]]; then
    WITH_LITELLM=true
fi

echo -e "${BLUE}Starting SAP AI Core OpenAI Proxy...${NC}"

# Check if key.json exists
if [ ! -f "key.json" ]; then
    echo -e "${RED}Error: key.json not found. Please create it with your SAP AI Core credentials.${NC}"
    exit 1
fi

# Kill existing processes
echo -e "${BLUE}Cleaning up existing processes...${NC}"
lsof -ti:4000 | xargs kill -9 2>/dev/null || true
lsof -ti:8082 | xargs kill -9 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Start SAP OAuth Proxy
echo -e "${GREEN}Starting SAP AI Core Proxy on port 4000...${NC}"
uv run python src/sap_openai_proxy.py > logs/sap_proxy.log 2>&1 &
SAP_PROXY_PID=$!
echo "SAP Proxy PID: $SAP_PROXY_PID"

# Wait for SAP proxy to start
sleep 3

# Check if SAP proxy is running
if ! curl -s http://localhost:4000/health > /dev/null 2>&1; then
    echo -e "${RED}Failed to start SAP AI Core Proxy${NC}"
    cat logs/sap_proxy.log
    exit 1
fi
echo -e "${GREEN}SAP AI Core Proxy started successfully${NC}"

# Start LiteLLM if requested
if [ "$WITH_LITELLM" = true ]; then
    echo -e "${YELLOW}Starting optional LiteLLM layer on port 8082...${NC}"
    uv run litellm --config config/litellm_config.yaml --port 8082 > logs/litellm_proxy.log 2>&1 &
    LITELLM_PID=$!
    echo "LiteLLM PID: $LITELLM_PID"

    sleep 5

    if ! curl -s http://localhost:8082/health > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: LiteLLM health check failed, but service may still be starting...${NC}"
    fi
    echo -e "${GREEN}LiteLLM Proxy started successfully${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
if [ "$WITH_LITELLM" = true ]; then
    echo -e "${GREEN}Both services are running!${NC}"
else
    echo -e "${GREEN}SAP AI Core Proxy is running!${NC}"
fi
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}Primary Service (Recommended):${NC}"
echo "  URL: http://localhost:4000/v1"
echo "  Features: Full OpenAI compatibility + Streaming ✅"
echo ""

if [ "$WITH_LITELLM" = true ]; then
    echo -e "${YELLOW}Optional LiteLLM Layer:${NC}"
    echo "  URL: http://localhost:8082/v1"
    echo "  Features: Load balancing, retries, rate limiting"
    echo "  Note: Streaming not supported ⚠️"
    echo ""
fi

echo "Process IDs:"
echo "  SAP Proxy:  $SAP_PROXY_PID"
if [ "$WITH_LITELLM" = true ]; then
    echo "  LiteLLM:    $LITELLM_PID"
fi
echo ""
echo "Logs:"
echo "  SAP Proxy:  logs/sap_proxy.log"
if [ "$WITH_LITELLM" = true ]; then
    echo "  LiteLLM:    logs/litellm_proxy.log"
fi
echo ""
echo "To stop services, run: scripts/stop.sh"
echo ""
echo -e "${BLUE}Test with:${NC}"
echo 'curl -X POST http://localhost:4000/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'"'"
echo ""
if [ "$WITH_LITELLM" = false ]; then
    echo -e "${YELLOW}Tip: To start with LiteLLM, run: ./scripts/start.sh --with-litellm${NC}"
    echo ""
fi

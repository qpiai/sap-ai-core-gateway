#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${RED}Stopping SAP AI Core OpenAI Proxy services...${NC}"

# Kill processes on ports 4000 and 8082
if lsof -ti:4000 > /dev/null 2>&1; then
    echo "Stopping SAP OAuth Proxy (port 4000)..."
    lsof -ti:4000 | xargs kill -9
    echo -e "${GREEN}SAP OAuth Proxy stopped${NC}"
else
    echo "SAP OAuth Proxy is not running"
fi

if lsof -ti:8082 > /dev/null 2>&1; then
    echo "Stopping LiteLLM Proxy (port 8082)..."
    lsof -ti:8082 | xargs kill -9
    echo -e "${GREEN}LiteLLM Proxy stopped${NC}"
else
    echo "LiteLLM Proxy is not running"
fi

echo -e "${GREEN}All services stopped${NC}"

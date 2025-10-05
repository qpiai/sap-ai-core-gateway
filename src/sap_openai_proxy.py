#!/usr/bin/env python3
"""
Simple OpenAI-compatible proxy for SAP AI Core
This is a minimal FastAPI app that LiteLLM can use as an OpenAI backend
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import json

# Import our utils
from utils.sap_auth import SAPAuth
from utils.sap_client import SAPClient
from utils.deployment_cache import DeploymentCache

# Initialize SAP components
auth = SAPAuth()
client = SAPClient(auth)
cache = DeploymentCache(client)

# FastAPI app
app = FastAPI(title="SAP AI Core OpenAI Proxy")

# API Key from environment
EXPECTED_API_KEY = os.getenv("SAP_PROXY_API_KEY", "sk-sap-proxy-secret-key-2024")


def verify_api_key(authorization: str = Header(None)):
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Support both "Bearer <key>" and just "<key>"
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict]  # Support both string and vision format (array of content parts)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    n: Optional[int] = 1
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    try:
        models = cache.get_available_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "sap-ai-core"
                }
                for model in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible chat completions endpoint"""
    try:

        # Get deployment ID
        deployment_id = cache.get_deployment_id(request.model)

        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Call SAP API
        response = await client.chat_completion(
            deployment_id=deployment_id,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            n=request.n,
            stop=request.stop,
            stream=request.stream,
            tools=request.tools,
            tool_choice=request.tool_choice
        )

        # If streaming, SAP should return a streaming response
        if request.stream:
            # Wrapper to ensure the async generator is properly consumed
            async def generate():
                async for chunk in response:
                    yield chunk

            # Return the streaming response with proper headers
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )

        # SAP returns OpenAI-compatible format, return as-is
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("SAP_PROXY_PORT", "4000"))

    print("=" * 80)
    print("SAP AI Core OpenAI-Compatible Proxy")
    print("=" * 80)
    print(f"\n🚀 Starting on http://0.0.0.0:{port}")
    print(f"📋 Available models: {cache.get_available_models()}")
    print("\n" + "=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=port)

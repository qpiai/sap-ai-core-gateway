"""
SAP AI Core API Client
Wraps SAP AI Core inference API calls
"""

import os
import json
import httpx
from typing import Dict, Any, List
from .sap_auth import SAPAuth


class SAPClient:
    """Client for SAP AI Core inference API"""

    def __init__(self, auth: SAPAuth):
        """Initialize with auth handler"""
        self.auth = auth
        self.api_url = None
        self.resource_group = os.getenv("SAP_RESOURCE_GROUP", "default")

        # Load API URL from key.json or env
        key_file = os.getenv("SAP_KEY_FILE", "key.json")
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                creds = json.load(f)
                self.api_url = creds.get("serviceurls", {}).get("AI_API_URL")

        # Override with env var if present
        self.api_url = os.getenv("SAP_API_URL", self.api_url)

        if not self.api_url:
            raise ValueError("Missing SAP_API_URL. Set in environment or provide key.json")

    async def _stream_completion(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float
    ):
        """Stream completion responses as async generator"""
        import httpx

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"SAP API error: {response.status_code} - {error_text.decode()}")

                async for line in response.aiter_lines():
                    if line:
                        # SSE format requires double newline between events
                        yield line + "\n\n"

    async def chat_completion(
        self,
        deployment_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Call SAP AI Core chat completion endpoint

        Returns:
            Dict if stream=False, AsyncGenerator if stream=True
        """
        token = self.auth.get_token()

        url = (
            f"{self.api_url}/v2/inference/deployments/{deployment_id}/"
            f"chat/completions?api-version=2023-05-15"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "n": kwargs.get("n", 1),
        }

        # Add optional parameters
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]

        # Add stream parameter
        if kwargs.get("stream", False):
            payload["stream"] = True

        # Add function/tool calling parameters (OpenAI format)
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]

        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            payload["tool_choice"] = kwargs["tool_choice"]

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Handle streaming
        if kwargs.get("stream", False):
            return self._stream_completion(
                url,
                headers,
                payload,
                kwargs.get("timeout", 120.0)
            )
        else:
            # Non-streaming request
            import httpx
            async with httpx.AsyncClient() as aclient:
                response = await aclient.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=kwargs.get("timeout", 120.0)
                )

                if response.status_code != 200:
                    raise Exception(f"SAP API error: {response.status_code} - {response.text}")

                return response.json()

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        token = self.auth.get_token()

        url = f"{self.api_url}/v2/lm/deployments"

        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json"
        }

        response = httpx.get(url, headers=headers, timeout=30.0)

        if response.status_code != 200:
            raise Exception(f"Failed to list deployments: {response.status_code} - {response.text}")

        return response.json().get("resources", [])

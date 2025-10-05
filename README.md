# SAP AI Core OpenAI Proxy

OpenAI-compatible proxy for SAP AI Core. Handles OAuth2 authentication so you can use SAP AI Core with any OpenAI SDK.

## Why?

SAP AI Core is already OpenAI-compatible but requires OAuth2 authentication. This proxy translates simple API keys → OAuth2 tokens, letting you use standard OpenAI tools.

## Enterprise Use Cases

- **RAG Applications:** Use with any OpenAI-compatible framework to access SAP HANA Vector Engine
- **AI Agents:** Integrate SAP-managed LLMs with OpenAI-compatible agent frameworks
- **Compliance:** Keep LLM traffic within SAP infrastructure for regulated industries
- **Multi-vendor AI:** Standardize on OpenAI API across Azure, AWS Bedrock, and SAP AI Core

## Quick Start

**1. Install dependencies:**
```bash
uv sync
```

**2. Configure SAP credentials in `key.json`:**
```json
{
  "clientid": "your-client-id",
  "clientsecret": "your-client-secret",
  "url": "https://your-instance.authentication.region.hana.ondemand.com",
  "serviceurls": {
    "AI_API_URL": "https://api.ai.region.aws.ml.hana.ondemand.com"
  }
}
```

**3. Start the proxy:**
```bash
./scripts/start.sh
```

**4. Use with OpenAI SDK:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-sap-proxy-secret-key-2024"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming works perfectly
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## Features

- ✅ Full OpenAI API compatibility (chat, streaming, tools, vision)
- ✅ Automatic OAuth2 token management
- ✅ Model auto-discovery from SAP deployments
- ✅ API key authentication
- ✅ Docker support

## Docker

```bash
docker-compose up -d
```

## Configuration

Set in `.env` or environment:

| Variable | Default |
|----------|---------|
| `SAP_PROXY_API_KEY` | `sk-sap-proxy-secret-key-2024` |
| `SAP_CONFIG_PATH` | `key.json` |
| `SAP_PROXY_PORT` | `4000` |

## Advanced: LiteLLM Layer (Optional)

Start with LiteLLM for load balancing/retries:
```bash
./scripts/start.sh --with-litellm
```

Use port 8082 instead of 4000. **Note:** Streaming doesn't work through LiteLLM due to response buffering.

## Testing

```bash
# Run tests
uv run pytest tests/ -v

# Run examples
uv run python examples/example_usage.py
```

## Examples

**Streaming:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
```

**Function Calling:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }]
)
```

**Vision:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

## Architecture

```
Client → SAP Proxy (4000) → SAP AI Core
         [API Key]          [OAuth2]

Optional:
Client → LiteLLM (8082) → SAP Proxy (4000) → SAP AI Core
```

## License

MIT

---

Built with ❤️ by [QpiAI](https://qpiai.tech)

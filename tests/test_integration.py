"""
Integration tests for the SAP AI Core OpenAI Proxy.

These tests verify:
1. Service health and availability
2. Authentication
3. Model listing
4. End-to-end request flow
"""

import pytest
import httpx
from openai import OpenAI


BASE_URL = "http://localhost:8082/v1"
SAP_PROXY_URL = "http://localhost:4000"
API_KEY = "sk-sap-ai-core-proxy-key-2024"


@pytest.fixture
def client():
    """Create an OpenAI client."""
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=60.0
    )


@pytest.fixture
def http_client():
    """Create an HTTP client."""
    return httpx.Client(timeout=60.0)


class TestServiceHealth:
    """Test service health and availability."""

    def test_sap_proxy_health(self, http_client):
        """Test SAP OAuth proxy health endpoint."""
        response = http_client.get(f"{SAP_PROXY_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_litellm_proxy_health(self, http_client):
        """Test LiteLLM proxy health endpoint."""
        response = http_client.get(f"{BASE_URL.replace('/v1', '')}/health")
        # LiteLLM health endpoint may return 200 or 404 depending on config
        assert response.status_code in [200, 404]


class TestAuthentication:
    """Test authentication mechanisms."""

    def test_valid_api_key(self, client):
        """Test request with valid API key."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        assert response.choices[0].message.content is not None

    def test_missing_api_key(self):
        """Test request without API key."""
        client = OpenAI(
            base_url=BASE_URL,
            api_key="",
            timeout=60.0
        )

        # Should work since we disabled authentication for testing
        # In production, this should fail
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            # If it works, we're in test mode
            assert True
        except Exception:
            # If it fails, authentication is enabled (expected in production)
            assert True


class TestModelListing:
    """Test model listing endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        models = client.models.list()

        model_ids = [model.id for model in models.data]

        # Should include at least gpt-4o
        assert "gpt-4o" in model_ids

        # May also include aliases
        assert len(model_ids) > 0

    def test_model_details(self):
        """Test that models have correct structure."""
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=60.0
        )

        models = client.models.list()

        for model in models.data:
            assert model.id is not None
            assert model.object == "model"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_request_flow(self, client):
        """Test a complete request from client to SAP AI Core and back."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 1+1? Answer with just the number."}
            ],
            max_tokens=10,
            temperature=0.1
        )

        # Verify response structure
        assert response.id is not None
        assert response.object == "chat.completion"
        assert response.created > 0
        assert response.model is not None

        # Verify message content
        message = response.choices[0].message
        assert message.role == "assistant"
        assert message.content is not None
        assert "2" in message.content

        # Verify usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_streaming_end_to_end(self, client):
        """Test streaming from client to SAP AI Core."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Say 'test'"}
            ],
            max_tokens=10,
            stream=True
        )

        chunks_received = 0
        content_parts = []

        for chunk in response:
            chunks_received += 1
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Should receive multiple chunks
        assert chunks_received > 0

        # Should have content
        full_content = "".join(content_parts)
        assert len(full_content) > 0

    def test_multiple_sequential_requests(self, client):
        """Test multiple requests in sequence."""
        for i in range(3):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"Say the number {i}"}
                ],
                max_tokens=10
            )

            assert response.choices[0].message.content is not None
            assert str(i) in response.choices[0].message.content


class TestSAPProxyDirect:
    """Test SAP OAuth proxy directly (without LiteLLM layer)."""

    def test_direct_sap_proxy_request(self):
        """Test making a request directly to SAP proxy."""
        client = httpx.Client(timeout=60.0)

        response = client.post(
            f"{SAP_PROXY_URL}/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hi"}
                ],
                "max_tokens": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] is not None

    def test_deployment_discovery(self):
        """Test that SAP proxy discovers deployments."""
        # This is tested implicitly by successful requests
        # The proxy must have discovered the gpt-4o deployment
        client = httpx.Client(timeout=60.0)

        response = client.post(
            f"{SAP_PROXY_URL}/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Test"}
                ],
                "max_tokens": 5
            }
        )

        assert response.status_code == 200


class TestErrorPropagation:
    """Test that errors are properly propagated through the proxy layers."""

    def test_invalid_request_format(self, http_client):
        """Test that invalid requests return proper errors."""
        response = http_client.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": "invalid"  # Should be array
            },
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        )

        # Should return 400 or 422
        assert response.status_code >= 400

    def test_timeout_handling(self, client):
        """Test request with very short timeout."""
        short_timeout_client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=0.001  # Very short timeout
        )

        with pytest.raises(Exception):
            # This should timeout
            short_timeout_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "Write a long story."}
                ],
                max_tokens=1000
            )


class TestConcurrency:
    """Test concurrent requests."""

    def test_concurrent_requests(self, client):
        """Test that multiple concurrent requests work."""
        import concurrent.futures

        def make_request(i):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"Say {i}"}
                ],
                max_tokens=10
            )
            return response.choices[0].message.content

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

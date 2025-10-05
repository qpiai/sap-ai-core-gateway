"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest
import httpx
import time
from openai import OpenAI


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "streaming: Streaming tests")
    config.addinivalue_line("markers", "multiturn: Multi-turn conversation tests")


@pytest.fixture(scope="session", autouse=True)
def check_services_running():
    """
    Check that required services are running before tests start.
    This fixture runs once per test session.
    """
    sap_proxy_url = "http://localhost:4000"
    litellm_proxy_url = "http://localhost:8082"

    max_retries = 3
    retry_delay = 2

    # Check SAP OAuth Proxy
    for i in range(max_retries):
        try:
            response = httpx.get(f"{sap_proxy_url}/health", timeout=5.0)
            if response.status_code == 200:
                print(f"\n✓ SAP OAuth Proxy is running on {sap_proxy_url}")
                break
        except Exception as e:
            if i == max_retries - 1:
                pytest.exit(
                    f"SAP OAuth Proxy is not running on {sap_proxy_url}. "
                    f"Please start it with: ./scripts/start.sh"
                )
            time.sleep(retry_delay)

    # Check LiteLLM Proxy
    for i in range(max_retries):
        try:
            # Try health endpoint or just check if service responds
            response = httpx.get(f"{litellm_proxy_url}/health", timeout=5.0)
            print(f"\n✓ LiteLLM Proxy is running on {litellm_proxy_url}")
            break
        except Exception:
            # LiteLLM might not have /health, try a different endpoint
            try:
                response = httpx.get(f"{litellm_proxy_url}/v1/models", timeout=5.0)
                if response.status_code in [200, 401, 403]:
                    print(f"\n✓ LiteLLM Proxy is running on {litellm_proxy_url}")
                    break
            except Exception as e:
                if i == max_retries - 1:
                    pytest.exit(
                        f"LiteLLM Proxy is not running on {litellm_proxy_url}. "
                        f"Please start it with: ./scripts/start.sh"
                    )
                time.sleep(retry_delay)


@pytest.fixture(scope="session")
def base_url():
    """Base URL for the LiteLLM proxy."""
    return "http://localhost:8082/v1"


@pytest.fixture(scope="session")
def sap_proxy_url():
    """Base URL for the SAP OAuth proxy."""
    return "http://localhost:4000"


@pytest.fixture(scope="session")
def api_key():
    """API key for authentication."""
    return "sk-sap-ai-core-proxy-key-2024"


@pytest.fixture(scope="session")
def default_model():
    """Default model to use for tests."""
    return "gpt-4o"


@pytest.fixture
def openai_client(base_url, api_key):
    """
    Create an OpenAI client for testing.
    This fixture is function-scoped, so a new client is created for each test.
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0
    )


@pytest.fixture
def http_client(api_key):
    """
    Create an HTTP client for direct API testing.
    This fixture is function-scoped.
    """
    return httpx.Client(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        timeout=60.0
    )


@pytest.fixture
def simple_messages():
    """A simple message list for testing."""
    return [
        {"role": "user", "content": "Hello"}
    ]


@pytest.fixture
def system_messages():
    """Message list with system message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]


@pytest.fixture
def conversation_history():
    """A multi-turn conversation history."""
    return [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"}
    ]


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test names.
    """
    for item in items:
        # Add markers based on test class or function names
        if "stream" in item.nodeid.lower():
            item.add_marker(pytest.mark.streaming)

        if "multi_turn" in item.nodeid.lower() or "MultiTurn" in item.nodeid:
            item.add_marker(pytest.mark.multiturn)

        if "integration" in item.nodeid.lower() or "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        if "concurrent" in item.nodeid.lower() or "long" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

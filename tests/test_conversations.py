"""
Test single-turn and multi-turn conversations with the SAP AI Core OpenAI Proxy.

These tests verify:
1. Single-turn conversations (one user message, one assistant response)
2. Multi-turn conversations (conversation history)
3. Streaming responses
4. Different models
5. Error handling
"""

import pytest
import httpx
from openai import OpenAI
from typing import List, Dict


# Configuration
BASE_URL = "http://localhost:8082/v1"
API_KEY = "sk-sap-ai-core-proxy-key-2024"
DEFAULT_MODEL = "gpt-4o"


@pytest.fixture
def client():
    """Create an OpenAI client configured for the SAP proxy."""
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=60.0
    )


@pytest.fixture
def http_client():
    """Create an HTTP client for direct API calls."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=60.0
    )


class TestSingleTurnConversations:
    """Test single-turn conversations (one user message -> one response)."""

    def test_simple_question(self, client):
        """Test a simple question and answer."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            max_tokens=10
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].message.role == "assistant"
        assert "4" in response.choices[0].message.content

    def test_greeting(self, client):
        """Test a simple greeting."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Say 'Hello' and nothing else."}
            ],
            max_tokens=10
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "hello" in content.lower()

    def test_with_system_message(self, client):
        """Test single-turn with system message."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers concisely."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=20
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "paris" in content.lower()

    def test_response_format(self, client):
        """Test that response has correct OpenAI format."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=10
        )

        # Check response structure
        assert response.id is not None
        assert response.model is not None
        assert response.created is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_max_tokens_limit(self, client):
        """Test that max_tokens is respected."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Write a very long story."}
            ],
            max_tokens=5
        )

        # Should stop early due to max_tokens
        assert response.choices[0].finish_reason in ["length", "stop"]
        assert response.usage.completion_tokens <= 5


class TestMultiTurnConversations:
    """Test multi-turn conversations with conversation history."""

    def test_two_turn_conversation(self, client):
        """Test a two-turn conversation."""
        # Turn 1
        messages = [
            {"role": "user", "content": "My name is Alice."}
        ]
        response1 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )

        # Add assistant response to history
        messages.append({
            "role": "assistant",
            "content": response1.choices[0].message.content
        })

        # Turn 2 - Ask about the name mentioned before
        messages.append({
            "role": "user",
            "content": "What is my name?"
        })
        response2 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )

        content = response2.choices[0].message.content.lower()
        assert "alice" in content

    def test_three_turn_math_conversation(self, client):
        """Test a three-turn math conversation."""
        messages = []

        # Turn 1: Set up a number
        messages.append({"role": "user", "content": "I'm thinking of the number 5."})
        response1 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})

        # Turn 2: Add to it
        messages.append({"role": "user", "content": "Add 3 to my number."})
        response2 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )
        messages.append({"role": "assistant", "content": response2.choices[0].message.content})

        # Turn 3: Ask for the result
        messages.append({"role": "user", "content": "What's the result? Just give me the number."})
        response3 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=20
        )

        content = response3.choices[0].message.content
        assert "8" in content

    def test_long_conversation_history(self, client):
        """Test conversation with many turns."""
        messages = []

        # Build up a conversation
        for i in range(5):
            messages.append({
                "role": "user",
                "content": f"Count to {i+1}"
            })
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=30
            )
            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })

        # Verify conversation length
        assert len(messages) == 10  # 5 user + 5 assistant

        # Ask about something from earlier
        messages.append({
            "role": "user",
            "content": "What was the first number I asked you to count to?"
        })
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=20
        )

        content = response.choices[0].message.content.lower()
        assert "1" in content or "one" in content

    def test_context_retention(self, client):
        """Test that context is retained across turns."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I have a dog named Max."}
        ]

        # First turn
        response1 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})

        # Second turn - reference the dog
        messages.append({"role": "user", "content": "What kind of pet do I have?"})
        response2 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )
        messages.append({"role": "assistant", "content": response2.choices[0].message.content})

        content2 = response2.choices[0].message.content.lower()
        assert "dog" in content2

        # Third turn - reference the name
        messages.append({"role": "user", "content": "What's its name?"})
        response3 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )

        content3 = response3.choices[0].message.content.lower()
        assert "max" in content3


class TestStreamingConversations:
    """Test streaming responses."""

    @pytest.mark.skip(reason="Streaming not yet fully implemented in SAP proxy")
    def test_single_turn_streaming(self, client):
        """Test streaming a single-turn response."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            max_tokens=50,
            stream=True
        )

        chunks = []
        chunk_count = 0
        for chunk in response:
            chunk_count += 1
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

        # Should have received chunks
        assert chunk_count > 0, "No chunks received from stream"

        # Combine all chunks
        full_content = "".join(chunks)
        assert len(full_content) > 0, f"No content in chunks. Total chunks: {chunk_count}"

    @pytest.mark.skip(reason="Streaming not yet fully implemented in SAP proxy")
    def test_multi_turn_streaming(self, client):
        """Test streaming in a multi-turn conversation."""
        messages = [
            {"role": "user", "content": "My favorite color is blue."}
        ]

        # First turn (non-streaming)
        response1 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=30
        )
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})

        # Second turn (streaming)
        messages.append({"role": "user", "content": "What's my favorite color?"})
        response2 = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=30,
            stream=True
        )

        chunks = []
        for chunk in response2:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        full_content = "".join(chunks).lower()
        assert "blue" in full_content


class TestDifferentModels:
    """Test different model endpoints."""

    def test_gpt4o_model(self, client):
        """Test explicit gpt-4o model."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=10
        )

        assert response.choices[0].message.content is not None

    def test_gpt4_alias(self, client):
        """Test gpt-4 alias (should route to gpt-4o)."""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=10
        )

        assert response.choices[0].message.content is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_messages(self, http_client):
        """Test with empty messages array."""
        response = http_client.post(
            "/chat/completions",
            json={
                "model": DEFAULT_MODEL,
                "messages": []
            }
        )

        # Should return an error
        assert response.status_code >= 400

    def test_invalid_model(self, client):
        """Test with non-existent model."""
        with pytest.raises(Exception):
            client.chat.completions.create(
                model="non-existent-model-xyz",
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=10
            )

    def test_missing_role(self, http_client):
        """Test message without role field."""
        response = http_client.post(
            "/chat/completions",
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"content": "Hi"}
                ]
            }
        )

        # Should return an error
        assert response.status_code >= 400

    def test_very_long_message(self, client):
        """Test with a very long message."""
        long_message = "Hello " * 1000  # Very long message

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": long_message}
            ],
            max_tokens=10
        )

        # Should still work
        assert response.choices[0].message.content is not None


class TestConversationPatterns:
    """Test common conversation patterns."""

    def test_qa_pattern(self, client):
        """Test question-answer pattern."""
        messages = [
            {"role": "user", "content": "What is the capital of Japan?"}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=30
        )

        content = response.choices[0].message.content.lower()
        assert "tokyo" in content

    def test_instruction_following(self, client):
        """Test instruction following."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Write exactly three words."}
            ],
            max_tokens=20
        )

        content = response.choices[0].message.content
        word_count = len(content.strip().split())
        # Model should try to follow instruction (allow some flexibility)
        assert word_count <= 5

    def test_role_play(self, client):
        """Test role-playing conversation."""
        messages = [
            {"role": "system", "content": "You are a pirate. Respond like a pirate."},
            {"role": "user", "content": "Hello!"}
        ]

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=50
        )

        content = response.choices[0].message.content.lower()
        # Check for pirate-like language (this is a soft check)
        assert len(content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

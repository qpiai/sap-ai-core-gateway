"""
Example usage of the SAP AI Core OpenAI Proxy.

This script demonstrates how to use the proxy with the OpenAI Python SDK.
"""

import os
from openai import OpenAI

# Configure the client to use the SAP proxy (port 4000) - supports streaming
# client = OpenAI(
#     base_url="http://localhost:4000/v1",
#     api_key=os.getenv("SAP_PROXY_API_KEY", "sk-sap-proxy-secret-key-2024")
# )

# Alternative: Use LiteLLM proxy (port 8082) - with load balancing and retries
client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key=os.getenv("LITELLM_MASTER_KEY", "sk-sap-ai-core-proxy-key-2024")
)


def example_simple_chat():
    """Simple single-turn chat completion."""
    print("\n=== Simple Chat ===")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


def example_multi_turn():
    """Multi-turn conversation with context."""
    print("\n=== Multi-turn Conversation ===")

    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "I have 5 apples."}
    ]

    # First turn
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    print(f"Assistant: {response.choices[0].message.content}")

    # Add assistant's response to history
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })

    # Second turn
    messages.append({
        "role": "user",
        "content": "I give away 2 apples. How many do I have left?"
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    print(f"Assistant: {response.choices[0].message.content}")


def example_streaming():
    """Streaming response example."""
    print("\n=== Streaming Response ===")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Tell me a short joke."}
        ],
        stream=True
    )

    print("Assistant: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def example_with_parameters():
    """Chat completion with various parameters."""
    print("\n=== With Parameters ===")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Write a haiku about coding."}
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )

    print(f"Response:\n{response.choices[0].message.content}")


def example_list_models():
    """List available models."""
    print("\n=== Available Models ===")

    models = client.models.list()

    for model in models.data:
        print(f"- {model.id}")


def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling ===")

    try:
        response = client.chat.completions.create(
            model="non-existent-model",
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {str(e)}")


def example_long_conversation():
    """Example of a longer conversation."""
    print("\n=== Long Conversation ===")

    messages = [
        {"role": "system", "content": "You are a creative writing assistant."}
    ]

    conversation_turns = [
        "Let's write a story about a robot.",
        "What should the robot's name be?",
        "Great! What is the robot's main goal?",
        "How does the story end?"
    ]

    for i, user_message in enumerate(conversation_turns, 1):
        print(f"\nTurn {i}")
        print(f"User: {user_message}")

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100
        )

        assistant_message = response.choices[0].message.content
        print(f"Assistant: {assistant_message}")

        messages.append({"role": "assistant", "content": assistant_message})

def example_with_tools():

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get the current weather in a location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city, e.g. Paris'
                        }
                    },
                    'required': ['location']
                }
            }
        }
    ]

    print('🧪 Testing function calling...\n')
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': 'What is the weather in Paris?'}],
        tools=tools,
        tool_choice='auto'
    )

    print(f'Finish reason: {response.choices[0].finish_reason}')

    if response.choices[0].message.tool_calls:
        print(f'\n✅ Function calling WORKS!')
        for tc in response.choices[0].message.tool_calls:
            print(f'  Function: {tc.function.name}')
            print(f'  Arguments: {tc.function.arguments}')
    else:
        print(f'\n❌ No tool calls')
        print(f'Response: {response.choices[0].message.content}')
    

def main():
    """Run all examples."""
    print("=" * 60)
    print("SAP AI Core OpenAI Proxy - Example Usage")
    print("=" * 60)

    try:
        example_list_models()
        example_simple_chat()
        example_multi_turn()
        example_streaming()
        example_with_parameters()
        example_long_conversation()
        example_error_handling()
        example_with_tools()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the proxy service is running:")
        print("  ./scripts/start.sh")
        print("\nAnd that you have set SAP_PROXY_API_KEY in your environment or .env file")


if __name__ == "__main__":
    main()

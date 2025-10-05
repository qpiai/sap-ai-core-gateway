"""
Langflow Language Model Component with SAP AI Core support (Standalone Version)
All dependencies are embedded - no external imports needed
"""
from typing import Any, Dict, Iterator, List, Optional
import json
import requests

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langflow.base.models.anthropic_constants import ANTHROPIC_MODELS
from langflow.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS
from langflow.base.models.model import LCModelComponent
from langflow.base.models.openai_constants import OPENAI_MODEL_NAMES
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs.inputs import BoolInput, StrInput
from langflow.io import DropdownInput, MessageInput, MultilineInput, SecretStrInput, SliderInput
from langflow.schema.dotdict import dotdict


# ============================================================================
# SAP AI CORE CLIENT (Embedded)
# ============================================================================

class SAPAICore:
    """SAP AI Core client for OAuth2 authentication and API calls"""

    def __init__(self, credentials: Dict[str, Any], resource_group: str = 'default'):
        self.client_id = credentials['clientid']
        self.client_secret = credentials['clientsecret']
        self.auth_url = credentials['url']
        self.api_url = credentials['serviceurls']['AI_API_URL']
        self.resource_group = resource_group
        self.access_token = None
        # Use a session to reuse connections and avoid "too many open files"
        self.session = requests.Session()

    def get_access_token(self) -> str:
        """Get OAuth2 access token using client credentials"""
        token_url = f"{self.auth_url}/oauth/token"

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = self.session.post(token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data['access_token']
        return self.access_token

    def chat_completion(self, deployment_id: str, messages: list,
                       max_tokens: int = 1000, temperature: float = 0.7,
                       tools: Optional[List[Dict]] = None) -> Dict:
        """Create chat completion request to AI model"""
        if not self.access_token:
            self.get_access_token()

        url = f"{self.api_url}/v2/inference/deployments/{deployment_id}/chat/completions?api-version=2023-05-15"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'AI-Resource-Group': self.resource_group,
            'Content-Type': 'application/json'
        }

        payload = {
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        # Add tools if provided (for function calling)
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'

        response = self.session.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def list_deployments(self) -> Dict:
        """List all available AI model deployments"""
        if not self.access_token:
            self.get_access_token()

        url = f"{self.api_url}/v2/lm/deployments"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'AI-Resource-Group': self.resource_group,
            'Content-Type': 'application/json'
        }

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the session to free up resources"""
        if hasattr(self, 'session'):
            self.session.close()


# ============================================================================
# LANGCHAIN WRAPPER (Embedded)
# ============================================================================

class ChatSAPAICore(BaseChatModel):
    """LangChain wrapper for SAP AI Core chat models"""

    sap_credentials: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    deployment_id: Optional[str] = Field(default=None)
    resource_group: str = Field(default="default")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    streaming: bool = Field(default=False)

    # Store tools as a private attribute to avoid serialization issues
    _tools: Optional[List[Any]] = None
    _client: Optional[SAPAICore] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize client after model is created"""
        super().model_post_init(__context)
        if self.sap_credentials and not self._client:
            self._client = SAPAICore(self.sap_credentials, self.resource_group)
            self._client.get_access_token()

            # Auto-detect deployment if not provided
            if not self.deployment_id:
                deployments = self._client.list_deployments()
                if deployments.get('resources'):
                    for deployment in deployments['resources']:
                        if deployment.get('status') == 'RUNNING':
                            self.deployment_id = deployment['id']
                            break

    @property
    def _llm_type(self) -> str:
        return "sap-ai-core"

    def _convert_messages_to_sap_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to SAP AI Core format"""
        sap_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                sap_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                sap_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                sap_messages.append({"role": "assistant", "content": message.content})
            else:
                sap_messages.append({"role": "user", "content": str(message.content)})
        return sap_messages

    def _convert_tools_to_openai_format(self) -> Optional[List[Dict]]:
        """Convert LangChain tools to OpenAI function calling format"""
        if not self._tools:
            return None

        openai_tools = []
        for tool in self._tools:
            # Handle different tool formats
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                # Get the args_schema - it might be a Pydantic model class
                args_schema = getattr(tool, 'args_schema', None)

                # If args_schema is a Pydantic model class, convert it to a dict
                if args_schema is not None:
                    if hasattr(args_schema, 'model_json_schema'):
                        # Pydantic v2
                        parameters = args_schema.model_json_schema()
                    elif hasattr(args_schema, 'schema'):
                        # Pydantic v1
                        parameters = args_schema.schema()
                    else:
                        parameters = {"type": "object", "properties": {}}
                else:
                    parameters = {"type": "object", "properties": {}}

                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters
                    }
                }
                openai_tools.append(tool_def)
        return openai_tools if openai_tools else None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using SAP AI Core"""
        if not self._client:
            raise ValueError("SAP AI Core client not initialized")

        sap_messages = self._convert_messages_to_sap_format(messages)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Convert tools to OpenAI format if present
        tools_formatted = self._convert_tools_to_openai_format()

        response = self._client.chat_completion(
            deployment_id=self.deployment_id,
            messages=sap_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools_formatted
        )

        if response.get('choices') and len(response['choices']) > 0:
            choice = response['choices'][0]
            message_data = choice['message']

            # Handle tool calls if present
            tool_calls = message_data.get('tool_calls', [])
            content = message_data.get('content', '')

            # Create AIMessage with tool calls
            if tool_calls:
                message = AIMessage(
                    content=content or '',
                    additional_kwargs={'tool_calls': tool_calls}
                )
            else:
                message = AIMessage(content=content)

            generation = ChatGeneration(
                message=message,
                generation_info={
                    "finish_reason": choice.get('finish_reason', 'stop'),
                    "model": response.get('model', 'unknown'),
                    "usage": response.get('usage', {})
                }
            )
            return ChatResult(generations=[generation])
        else:
            raise ValueError("No response from SAP AI Core")

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream (fallback to regular generation)"""
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "resource_group": self.resource_group,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @property
    def _llm_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for serialization"""
        return {
            "deployment_id": self.deployment_id,
            "resource_group": self.resource_group,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "_type": self._llm_type,
        }

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatSAPAICore":
        """Bind tools to the model for tool calling support"""
        # Create a copy with updated parameters, excluding credentials
        bound = self.model_copy(update=kwargs, deep=False)
        # Set the tools as a private attribute
        bound._tools = tools
        # Preserve the client and credentials from the original
        bound._client = self._client
        bound.sap_credentials = self.sap_credentials
        return bound


# ============================================================================
# SAP MODEL CONSTANTS (Embedded)
# ============================================================================

SAP_AI_CORE_MODELS = [
    # OpenAI Models (via Azure OpenAI)
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "o1",
    "o1-mini",
    "o3-mini",

    # Anthropic Claude Models (via AWS Bedrock)
    "anthropic--claude-3-5-sonnet",
    "anthropic--claude-3-7-sonnet",
    "anthropic--claude-3-opus",
    "anthropic--claude-3-sonnet",
    "anthropic--claude-3-haiku",

    # Google Gemini Models (via GCP Vertex AI)
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",

    # Amazon Nova Models (via AWS Bedrock)
    "amazon--nova-pro",
    "amazon--nova-lite",
    "amazon--nova-micro",

    # Meta Llama Models
    "meta--llama3-70b-instruct",
    "meta--llama3-1-70b-instruct",
    "meta--llama3-1-8b-instruct",
    "meta--llama3-2-1b-instruct",
    "meta--llama3-2-3b-instruct",

    # Mistral AI Models
    "mistralai--mistral-large",
    "mistralai--mixtral-8x7b-instruct-v01",
    "mistralai--mistral-small-3-1",

    # NVIDIA Models
    "nvidia--llama-3-2-nv-embedqa-1b",

    # Other Open Source Models
    "tiiuae--falcon-40b-instruct",
]


# ============================================================================
# LANGFLOW COMPONENT
# ============================================================================

class LanguageModelComponent(LCModelComponent):
    display_name = "Language Model"
    description = "Runs a language model given a specified provider (OpenAI, Anthropic, Google, SAP AI Core)."
    icon = "brain-circuit"
    category = "models"
    priority = 0

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Anthropic", "Google", "SAP AI Core"],
            value="OpenAI",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[
                {"icon": "OpenAI"},
                {"icon": "Anthropic"},
                {"icon": "Google"},
                {"icon": "Building"}
            ],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=OPENAI_MODEL_NAMES,
            value=OPENAI_MODEL_NAMES[0],
            info="Select the model to use",
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
        ),
        # SAP AI Core specific inputs
        StrInput(
            name="sap_client_id",
            display_name="SAP Client ID",
            info="SAP AI Core Client ID",
            required=False,
            show=False,
        ),
        SecretStrInput(
            name="sap_client_secret",
            display_name="SAP Client Secret",
            info="SAP AI Core Client Secret",
            required=False,
            show=False,
        ),
        StrInput(
            name="sap_auth_url",
            display_name="SAP Auth URL",
            info="SAP authentication URL",
            required=False,
            show=False,
        ),
        StrInput(
            name="sap_api_url",
            display_name="SAP API URL",
            info="SAP AI API URL",
            required=False,
            show=False,
        ),
        StrInput(
            name="sap_deployment_id",
            display_name="SAP Deployment ID",
            info="SAP AI Core deployment ID (leave empty to auto-detect)",
            required=False,
            show=False,
            advanced=True,
        ),
        StrInput(
            name="sap_resource_group",
            display_name="SAP Resource Group",
            info="SAP AI Core resource group name",
            value="default",
            required=False,
            show=False,
            advanced=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=True,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key,
            )

        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )

        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )

        if provider == "SAP AI Core":
            # Validate SAP credentials
            if not all([self.sap_client_id, self.sap_client_secret, self.sap_auth_url, self.sap_api_url]):
                msg = "All SAP credentials are required: Client ID, Client Secret, Auth URL, API URL"
                raise ValueError(msg)

            # Build credentials dictionary
            sap_credentials = {
                "clientid": self.sap_client_id,
                "clientsecret": self.sap_client_secret,
                "url": self.sap_auth_url,
                "serviceurls": {
                    "AI_API_URL": self.sap_api_url
                }
            }

            # Build SAP AI Core chat model
            return ChatSAPAICore(
                sap_credentials=sap_credentials,
                deployment_id=self.sap_deployment_id if self.sap_deployment_id else None,
                resource_group=self.sap_resource_group,
                temperature=temperature,
                streaming=stream,
            )

        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "OpenAI":
                build_config["model_name"]["options"] = OPENAI_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
                build_config["api_key"]["show"] = True
                build_config["api_key"]["required"] = True
                # Hide SAP fields
                build_config["sap_client_id"]["show"] = False
                build_config["sap_client_secret"]["show"] = False
                build_config["sap_auth_url"]["show"] = False
                build_config["sap_api_url"]["show"] = False
                build_config["sap_deployment_id"]["show"] = False
                build_config["sap_resource_group"]["show"] = False

            elif field_value == "Anthropic":
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["api_key"]["display_name"] = "Anthropic API Key"
                build_config["api_key"]["show"] = True
                build_config["api_key"]["required"] = True
                # Hide SAP fields
                build_config["sap_client_id"]["show"] = False
                build_config["sap_client_secret"]["show"] = False
                build_config["sap_auth_url"]["show"] = False
                build_config["sap_api_url"]["show"] = False
                build_config["sap_deployment_id"]["show"] = False
                build_config["sap_resource_group"]["show"] = False

            elif field_value == "Google":
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
                build_config["api_key"]["show"] = True
                build_config["api_key"]["required"] = True
                # Hide SAP fields
                build_config["sap_client_id"]["show"] = False
                build_config["sap_client_secret"]["show"] = False
                build_config["sap_auth_url"]["show"] = False
                build_config["sap_api_url"]["show"] = False
                build_config["sap_deployment_id"]["show"] = False
                build_config["sap_resource_group"]["show"] = False

            elif field_value == "SAP AI Core":
                build_config["model_name"]["options"] = SAP_AI_CORE_MODELS
                build_config["model_name"]["value"] = SAP_AI_CORE_MODELS[0]
                # Hide API key
                build_config["api_key"]["show"] = False
                build_config["api_key"]["required"] = False
                # Show SAP fields
                build_config["sap_client_id"]["show"] = True
                build_config["sap_client_id"]["required"] = True
                build_config["sap_client_secret"]["show"] = True
                build_config["sap_client_secret"]["required"] = True
                build_config["sap_auth_url"]["show"] = True
                build_config["sap_auth_url"]["required"] = True
                build_config["sap_api_url"]["show"] = True
                build_config["sap_api_url"]["required"] = True
                build_config["sap_deployment_id"]["show"] = True
                build_config["sap_resource_group"]["show"] = True

        return build_config

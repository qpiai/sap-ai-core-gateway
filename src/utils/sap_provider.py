"""
SAP AI Core Custom Provider for LiteLLM
Main integration point - handles all SAP-specific logic
"""

from typing import Dict, Any, List
from .sap_auth import SAPAuth
from .sap_client import SAPClient
from .deployment_cache import DeploymentCache


class SAPAICoreProvider:
    """Custom LiteLLM provider for SAP AI Core"""

    _instance = None

    def __new__(cls):
        """Singleton pattern - reuse auth/cache across requests"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize SAP components (only once)"""
        if self._initialized:
            return

        # Initialize auth, client, and cache
        self.auth = SAPAuth()
        self.client = SAPClient(self.auth)
        self.cache = DeploymentCache(self.client)

        self._initialized = True

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str = None,
        custom_llm_provider: str = None,
        print_verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle completion request for SAP AI Core
        This is called by LiteLLM
        """
        # Get deployment ID
        deployment_id = self.cache.get_deployment_id(model)

        if print_verbose:
            print(f"SAP AI Core: {model} → {deployment_id}")

        # Call SAP API
        response = self.client.chat_completion(
            deployment_id=deployment_id,
            messages=messages,
            **kwargs
        )

        # SAP returns OpenAI-compatible response, return as-is
        return response

"""
Deployment Discovery and Caching
Maps model names to SAP AI Core deployment IDs
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from .sap_client import SAPClient


class DeploymentCache:
    """Manages deployment discovery and model name mapping"""

    def __init__(self, client: SAPClient):
        """Initialize with SAP client"""
        self.client = client
        self.cache: Dict[str, Any] = {}
        self.cache_expiry: Optional[datetime] = None
        self.cache_ttl = int(os.getenv("SAP_DEPLOYMENT_CACHE_TTL", "300"))  # 5 minutes

    def get_deployment_id(self, model: str) -> str:
        """
        Get deployment ID for a model name
        Priority: env var > cache > API discovery
        """
        # 1. Check environment variable override
        env_var = f"SAP_MODEL_{model.upper().replace('-', '_').replace('.', '_')}"
        env_deployment = os.getenv(env_var)
        if env_deployment:
            return env_deployment

        # 2. Check if it's already a deployment ID (16 chars starting with 'd')
        if len(model) == 16 and model[0].isalpha():
            return model

        # 3. Check cache
        if self._is_cache_valid():
            for dep in self.cache.get("deployments", []):
                if self._matches_model(dep, model):
                    return dep["id"]

        # 4. Refresh cache and search again
        self._refresh_cache()

        for dep in self.cache.get("deployments", []):
            if self._matches_model(dep, model):
                return dep["id"]

        # Not found
        raise Exception(
            f"No deployment found for model '{model}'. "
            f"Available: {self.get_available_models()}"
        )

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.cache or not self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry

    def _refresh_cache(self):
        """Refresh deployment cache from SAP API"""
        deployments = self.client.list_deployments()

        # Filter running deployments only
        running = [d for d in deployments if d.get("status") == "RUNNING"]

        self.cache = {"deployments": running}
        self.cache_expiry = datetime.now() + timedelta(seconds=self.cache_ttl)

    def _matches_model(self, deployment: Dict[str, Any], model_name: str) -> bool:
        """Check if deployment matches model name"""
        # Extract model info from deployment
        details = deployment.get("details", {})
        resources = details.get("resources", {})
        backend = resources.get("backendDetails", {}) or resources.get("backend_details", {})
        model_info = backend.get("model", {})

        actual_name = model_info.get("name", "").lower()
        actual_version = model_info.get("version", "").lower()
        scenario = deployment.get("scenarioId", "").lower()

        model_lower = model_name.lower()

        # Match by model name
        if model_lower in actual_name:
            return True

        # Match by model-version
        if actual_version and f"{actual_name}-{actual_version}" == model_lower:
            return True

        # Match by scenario
        if model_lower in scenario:
            return True

        return False

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        if not self._is_cache_valid():
            self._refresh_cache()

        models = []
        for dep in self.cache.get("deployments", []):
            details = dep.get("details", {})
            resources = details.get("resources", {})
            backend = resources.get("backendDetails", {}) or resources.get("backend_details", {})
            model_info = backend.get("model", {})

            name = model_info.get("name", dep.get("scenarioId", "unknown"))
            version = model_info.get("version", "")

            if version:
                models.append(f"{name}-{version}")
            else:
                models.append(name)

        return models

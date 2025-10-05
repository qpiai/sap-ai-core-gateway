"""
SAP AI Core OAuth2 Authentication
Handles token acquisition, caching, and auto-refresh
"""

import os
import json
import httpx
from datetime import datetime, timedelta
from typing import Optional


class SAPAuth:
    """Manages OAuth2 authentication for SAP AI Core"""

    def __init__(self):
        """Initialize from key.json or environment variables"""
        self.client_id = None
        self.client_secret = None
        self.auth_url = None

        # Try to load from key.json
        key_file = os.getenv("SAP_KEY_FILE", "key.json")
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                creds = json.load(f)
                self.client_id = creds.get("clientid")
                self.client_secret = creds.get("clientsecret")
                self.auth_url = creds.get("url")

        # Override with environment variables if present
        self.client_id = os.getenv("SAP_CLIENT_ID", self.client_id)
        self.client_secret = os.getenv("SAP_CLIENT_SECRET", self.client_secret)
        self.auth_url = os.getenv("SAP_AUTH_URL", self.auth_url)

        # Validate
        if not all([self.client_id, self.client_secret, self.auth_url]):
            raise ValueError(
                "Missing SAP credentials. Set SAP_CLIENT_ID, SAP_CLIENT_SECRET, SAP_AUTH_URL "
                "or provide key.json"
            )

        # Token cache
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

    def get_token(self) -> str:
        """Get valid access token, refreshing if needed"""
        # Check cache
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                return self.access_token

        # Request new token
        token_url = f"{self.auth_url}/oauth/token"

        response = httpx.post(
            token_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=30.0
        )

        if response.status_code != 200:
            raise Exception(f"OAuth2 failed: {response.status_code} - {response.text}")

        token_data = response.json()
        self.access_token = token_data["access_token"]

        # Cache with 60s safety margin
        expires_in = token_data.get("expires_in", 3600)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)

        return self.access_token

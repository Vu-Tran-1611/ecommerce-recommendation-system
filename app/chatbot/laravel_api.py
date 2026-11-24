import os

import requests
from dotenv import load_dotenv

load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL")
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY", "")
REQUEST_TIMEOUT_SECONDS = 10


# Call a Laravel AI API endpoint with the service key
def laravel_get(path: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{LARAVEL_API_URL}{path}",
        params=params,
        headers={"X-AI-Service-Key": AI_SERVICE_KEY, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()

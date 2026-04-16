from __future__ import annotations

from typing import Any

import requests

from app.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
)


class OpenAIConfigurationError(RuntimeError):
    """Raised when the hosted LLM configuration is missing or unsupported."""


class OpenAIServiceError(RuntimeError):
    """Raised when the hosted LLM request fails."""


def call_openai_structured(
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> str:
    if LLM_PROVIDER != "openai":
        raise OpenAIConfigurationError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Set LLM_PROVIDER=openai."
        )

    if not OPENAI_API_KEY:
        raise OpenAIConfigurationError(
            "OPENAI_API_KEY is not set. Set it in the environment before starting the API."
        )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OpenAIServiceError("Could not reach the OpenAI API.") from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise OpenAIServiceError("OpenAI API returned a non-JSON response.") from exc

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAIServiceError("OpenAI API response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise OpenAIServiceError("OpenAI API response did not include a message.")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_chunks.append(text_value)
        if text_chunks:
            return "".join(text_chunks)

    raise OpenAIServiceError("OpenAI API response did not include structured text content.")

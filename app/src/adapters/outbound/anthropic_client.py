"""Shared helpers for adapters that call the Anthropic SDK directly."""

import json
import re
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message


def build_client(api_key: str, timeout: float, max_retries: int) -> AsyncAnthropic:
    """Create an AsyncAnthropic client with retry/timeout configuration.

    The SDK retries rate limits, connection errors, and 5xx responses with
    exponential backoff on its own — no extra retry layer is needed.
    """
    return AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)


def response_text(message: Message) -> str:
    """Concatenate the text blocks of a Messages API response."""
    return "".join(block.text for block in message.content if block.type == "text")


def parse_json_response(text: str) -> Any:
    """Parse a JSON payload from model output, tolerating markdown code fences.

    Raises:
        json.JSONDecodeError: If the payload is not valid JSON.
    """
    content = text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return json.loads(content)

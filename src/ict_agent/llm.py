"""LLM client helpers shared by the refactored runtime."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


KSYUN_BASE_URL = "https://kspmas.ksyun.com/v1/"
INFINI_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
SUPPORTED_PROVIDERS = ("auto", "ksyun", "infini")


def get_provider_help_text() -> str:
    return (
        "Available providers:\n"
        "  - ksyun  (default model: mco-4, env: KSYUN_API_KEY)\n"
        "  - infini (default model: deepseek-v3, env: INFINI_API_KEY)\n"
        "  - auto   (prefer ksyun, fall back to infini)"
    )


def import_openai_client():
    """Import the OpenAI SDK with a friendly error message."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("Error: Python package 'openai' is not installed in the current environment.")
        print("Install project dependencies first, then retry.")
        print("Example: pip install -e \".[dev]\"")
        sys.exit(1)
    return OpenAI


def create_client(provider: str = "ksyun") -> tuple["OpenAI", str, str, str]:
    """Create a client for the selected provider.

    Returns:
        (client, provider_name, default_model, base_url)
    """
    OpenAI = import_openai_client()

    provider = (provider or "ksyun").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        print(f"Error: unknown provider '{provider}'.")
        print(get_provider_help_text())
        sys.exit(1)

    if provider in ("ksyun", "auto"):
        if api_key := os.getenv("KSYUN_API_KEY"):
            base_url = os.getenv("KSYUN_BASE_URL", KSYUN_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), "ksyun", "mco-4", base_url
        if provider == "ksyun":
            print("Error: provider 'ksyun' selected but KSYUN_API_KEY is not set.")
            sys.exit(1)

    if provider in ("infini", "auto"):
        if api_key := os.getenv("INFINI_API_KEY"):
            base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), "infini", "deepseek-v3", base_url
        if provider == "infini":
            print("Error: provider 'infini' selected but INFINI_API_KEY is not set.")
            sys.exit(1)

    print("Error: no usable provider credentials found.")
    print(get_provider_help_text())
    sys.exit(1)


def list_models(client: "OpenAI") -> None:
    """Fetch and print available models from the provider."""
    print("Fetching available models...\n")
    try:
        models = client.models.list()
        model_list = sorted(models.data, key=lambda item: item.id)
        print(f"Found {len(model_list)} models:\n")
        for model in model_list:
            print(f"  - {model.id}")
        print()
    except Exception as exc:
        print(f"Failed to list models: {exc}")
        if "timed out" in str(exc).lower():
            print("Hint: provider is reachable enough to start the request, but the model listing call timed out.")
            print("You can still run the agent directly if you already know the model name.")
        sys.exit(1)

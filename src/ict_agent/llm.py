"""LLM client helpers shared by the refactored runtime."""

from __future__ import annotations

import os
import sys
from typing import Any


KSYUN_BASE_URL = "https://kspmas.ksyun.com/v1/"
KSYUN_ANTHROPIC_BASE_URL = "https://kspmas.ksyun.com"
INFINI_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
SUPPORTED_PROVIDERS = ("auto", "ksyun", "infini")

# Models that use the Anthropic Messages API (with prompt caching support).
# All other models use the OpenAI Chat Completions API.
ANTHROPIC_MODELS = frozenset({"mco-4", "mcs-1"})


def get_provider_help_text() -> str:
    return (
        "Available providers:\n"
        "  - ksyun  (default model: mco-4, env: KSYUN_API_KEY)\n"
        "  - infini (default model: deepseek-v3, env: INFINI_API_KEY)\n"
        "  - auto   (prefer ksyun, fall back to infini)"
    )


def is_anthropic_client(client: Any) -> bool:
    """Return True if *client* is an Anthropic SDK instance."""
    try:
        from anthropic import Anthropic
        return isinstance(client, Anthropic)
    except ImportError:
        return False


def is_anthropic_model(model: str) -> bool:
    """Return True if *model* should use the Anthropic Messages API."""
    return model in ANTHROPIC_MODELS


def _import_anthropic():
    """Import the Anthropic SDK with a friendly error message."""
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError:
        print("Error: Python package 'anthropic' is not installed.")
        print("Install it with: pip install anthropic")
        sys.exit(1)
    return Anthropic


def _import_openai():
    """Import the OpenAI SDK with a friendly error message."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("Error: Python package 'openai' is not installed in the current environment.")
        print("Install project dependencies first, then retry.")
        print("Example: pip install -e \".[dev]\"")
        sys.exit(1)
    return OpenAI


# Keep old name as alias for backward compat
import_openai_client = _import_openai


def create_client(provider: str = "ksyun") -> tuple[Any, str, str, str]:
    """Create a client for the selected provider.

    Returns:
        (client, provider_name, default_model, base_url)

    The *client* is either an ``openai.OpenAI`` or ``anthropic.Anthropic``
    instance depending on whether the default model uses the Anthropic
    Messages API.
    """
    provider = (provider or "ksyun").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        print(f"Error: unknown provider '{provider}'.")
        print(get_provider_help_text())
        sys.exit(1)

    if provider in ("ksyun", "auto"):
        if api_key := os.getenv("KSYUN_API_KEY"):
            default_model = "mco-4"
            # Claude models → Anthropic SDK for native Messages API + prompt caching
            if default_model in ANTHROPIC_MODELS:
                Anthropic = _import_anthropic()
                base_url = os.getenv("KSYUN_ANTHROPIC_BASE_URL", KSYUN_ANTHROPIC_BASE_URL)
                return Anthropic(api_key=api_key, base_url=base_url), "ksyun", default_model, base_url
            else:
                OpenAI = _import_openai()
                base_url = os.getenv("KSYUN_BASE_URL", KSYUN_BASE_URL)
                return OpenAI(api_key=api_key, base_url=base_url), "ksyun", default_model, base_url
        if provider == "ksyun":
            print("Error: provider 'ksyun' selected but KSYUN_API_KEY is not set.")
            sys.exit(1)

    if provider in ("infini", "auto"):
        if api_key := os.getenv("INFINI_API_KEY"):
            OpenAI = _import_openai()
            base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), "infini", "deepseek-v3", base_url
        if provider == "infini":
            print("Error: provider 'infini' selected but INFINI_API_KEY is not set.")
            sys.exit(1)

    print("Error: no usable provider credentials found.")
    print(get_provider_help_text())
    sys.exit(1)


def create_openai_client_for_model(provider: str, model: str) -> tuple[Any, str]:
    """Create an OpenAI-compatible client regardless of model.

    Used when a specific non-Anthropic path is needed (e.g. compactor with
    gpt-oss-120b while the main client is Anthropic).

    Returns:
        (client, base_url)
    """
    OpenAI = _import_openai()
    provider = (provider or "ksyun").strip().lower()
    if provider in ("ksyun", "auto"):
        if api_key := os.getenv("KSYUN_API_KEY"):
            base_url = os.getenv("KSYUN_BASE_URL", KSYUN_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), base_url
    if provider in ("infini", "auto"):
        if api_key := os.getenv("INFINI_API_KEY"):
            base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), base_url
    raise RuntimeError(f"No credentials for provider '{provider}'")


def list_models(client: Any) -> None:
    """Fetch and print available models from the provider."""
    if is_anthropic_client(client):
        print("Model listing is not supported for Anthropic clients.")
        print("Known models: " + ", ".join(sorted(ANTHROPIC_MODELS)))
        return
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

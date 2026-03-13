"""LLM client management — dual-SDK provider with per-model dispatch.

Ksyun hosts both Claude models (Anthropic Messages API) and OpenAI-compatible
models (gpt-oss-120b, deepseek-v3, etc.) under the same API key.  This module
manages both SDK clients and exposes a unified ``ModelRouter`` that returns the
correct client for any given model name.
"""

from __future__ import annotations

import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
#  Provider configuration
# ---------------------------------------------------------------------------

KSYUN_OPENAI_BASE_URL = "https://kspmas.ksyun.com/v1/"
KSYUN_ANTHROPIC_BASE_URL = "https://kspmas.ksyun.com"
INFINI_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
SUPPORTED_PROVIDERS = ("auto", "ksyun", "infini", "vllm")

# Models that use the Anthropic Messages API (with prompt caching support).
# All other models use the OpenAI Chat Completions API.
ANTHROPIC_MODELS = frozenset({"mco-4", "mcs-1", "mch-1", "mcs-5"})

# Keep legacy name for any external references
KSYUN_BASE_URL = KSYUN_OPENAI_BASE_URL


def get_provider_help_text() -> str:
    return (
        "Available providers:\n"
        "  - ksyun  (default model: mco-4, env: KSYUN_API_KEY)\n"
        "    Claude models (mco-4, mcs-1, mch-1): Anthropic Messages API with prompt caching\n"
        "    Other models (gpt-oss-120b, etc.): OpenAI Chat Completions API\n"
        "  - infini (default model: deepseek-v3, env: INFINI_API_KEY)\n"
        "  - vllm   (local vllm serve, env: VLLM_BASE_URL, VLLM_API_KEY optional)\n"
        "    OpenAI-compatible API at localhost (default: http://localhost:8000/v1)\n"
        "    Model name defaults to VLLM_MODEL or 'default'\n"
        "  - auto   (prefer ksyun, fall back to infini)"
    )


# ---------------------------------------------------------------------------
#  SDK imports
# ---------------------------------------------------------------------------

def _import_anthropic():
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError:
        print("Error: Python package 'anthropic' is not installed.")
        print("Install it with: pip install anthropic")
        sys.exit(1)
    return Anthropic


def _import_openai():
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("Error: Python package 'openai' is not installed.")
        print("Install project dependencies first: pip install -e \".[dev]\"")
        sys.exit(1)
    return OpenAI


# Keep old name as alias for backward compat
import_openai_client = _import_openai


# ---------------------------------------------------------------------------
#  Client type checks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  ModelRouter — the core abstraction
# ---------------------------------------------------------------------------

class ModelRouter:
    """Manages dual SDK clients and dispatches by model name.

    A single Ksyun API key can access both Claude models (Anthropic SDK) and
    OpenAI-compatible models (OpenAI SDK).  ``ModelRouter`` lazily creates
    clients on first use and caches them for reuse.

    Usage::

        router = ModelRouter("ksyun")
        client = router.client_for_model("mco-4")    # → Anthropic
        client = router.client_for_model("gpt-oss-120b")  # → OpenAI
    """

    def __init__(self, provider: str, api_key: str, default_model: str):
        self.provider = provider
        self.api_key = api_key
        self.default_model = default_model
        self._anthropic_client: Any = None
        self._openai_client: Any = None

    def client_for_model(self, model: str | None = None) -> Any:
        """Return the appropriate SDK client for *model*."""
        model = model or self.default_model
        if model in ANTHROPIC_MODELS:
            return self._get_anthropic()
        return self._get_openai()

    def _get_anthropic(self) -> Any:
        if self._anthropic_client is None:
            Anthropic = _import_anthropic()
            base_url = os.getenv("KSYUN_ANTHROPIC_BASE_URL", KSYUN_ANTHROPIC_BASE_URL)
            self._anthropic_client = Anthropic(api_key=self.api_key, base_url=base_url)
        return self._anthropic_client

    def _get_openai(self) -> Any:
        if self._openai_client is None:
            OpenAI = _import_openai()
            if self.provider == "vllm":
                base_url = os.getenv("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
            elif self.provider in ("ksyun", "auto"):
                base_url = os.getenv("KSYUN_BASE_URL", KSYUN_OPENAI_BASE_URL)
            elif self.provider == "infini":
                base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            else:
                base_url = KSYUN_OPENAI_BASE_URL
            self._openai_client = OpenAI(api_key=self.api_key, base_url=base_url)
        return self._openai_client

    @property
    def base_url(self) -> str:
        """Base URL for the default model's client (for display purposes)."""
        if self.default_model in ANTHROPIC_MODELS:
            return os.getenv("KSYUN_ANTHROPIC_BASE_URL", KSYUN_ANTHROPIC_BASE_URL)
        if self.provider == "vllm":
            return os.getenv("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
        if self.provider == "infini":
            return os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
        return os.getenv("KSYUN_BASE_URL", KSYUN_OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
#  create_client — backward-compatible entry point
# ---------------------------------------------------------------------------

def create_client(provider: str = "ksyun") -> tuple[Any, str, str, str]:
    """Create a client for the selected provider.

    Returns:
        (client_or_router, provider_name, default_model, base_url)

    For ksyun provider, returns a ``ModelRouter`` that can serve both
    Anthropic and OpenAI clients.  For infini, returns an OpenAI client
    directly.  Call sites should use ``get_client_for_model()`` to get
    the right client for a given model.
    """
    provider = (provider or "ksyun").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        print(f"Error: unknown provider '{provider}'.")
        print(get_provider_help_text())
        sys.exit(1)

    if provider in ("ksyun", "auto"):
        if api_key := os.getenv("KSYUN_API_KEY"):
            default_model = "mco-4"
            router = ModelRouter("ksyun", api_key, default_model)
            return router, "ksyun", default_model, router.base_url
        if provider == "ksyun":
            print("Error: provider 'ksyun' selected but KSYUN_API_KEY is not set.")
            sys.exit(1)

    if provider in ("infini", "auto"):
        if api_key := os.getenv("INFINI_API_KEY"):
            OpenAI = _import_openai()
            base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            # Wrap in router for consistency
            router = ModelRouter("infini", api_key, "deepseek-v3")
            return router, "infini", "deepseek-v3", base_url
        if provider == "infini":
            print("Error: provider 'infini' selected but INFINI_API_KEY is not set.")
            sys.exit(1)

    if provider == "vllm":
        # vllm serve exposes an OpenAI-compatible API on localhost.
        # VLLM_API_KEY is optional (vllm doesn't require auth by default).
        base_url = os.getenv("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        default_model = os.getenv("VLLM_MODEL", "default")
        router = ModelRouter("vllm", api_key, default_model)
        return router, "vllm", default_model, base_url

    print("Error: no usable provider credentials found.")
    print(get_provider_help_text())
    sys.exit(1)


def get_client_for_model(client_or_router: Any, model: str) -> Any:
    """Get the correct SDK client for *model*.

    If *client_or_router* is a ``ModelRouter``, dispatches by model name.
    Otherwise returns the client as-is (backward compat).
    """
    if isinstance(client_or_router, ModelRouter):
        return client_or_router.client_for_model(model)
    return client_or_router


# Legacy helper — kept for backward compat
def create_openai_client_for_model(provider: str, model: str) -> tuple[Any, str]:
    """Create an OpenAI-compatible client regardless of model."""
    OpenAI = _import_openai()
    provider = (provider or "ksyun").strip().lower()
    if provider in ("ksyun", "auto"):
        if api_key := os.getenv("KSYUN_API_KEY"):
            base_url = os.getenv("KSYUN_BASE_URL", KSYUN_OPENAI_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), base_url
    if provider in ("infini", "auto"):
        if api_key := os.getenv("INFINI_API_KEY"):
            base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
            return OpenAI(api_key=api_key, base_url=base_url), base_url
    if provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        return OpenAI(api_key=api_key, base_url=base_url), base_url
    raise RuntimeError(f"No credentials for provider '{provider}'")


def list_models(client_or_router: Any) -> None:
    """Fetch and print available models from the provider."""
    print("Known Anthropic models: " + ", ".join(sorted(ANTHROPIC_MODELS)))
    print()
    # Try listing OpenAI-compatible models
    if isinstance(client_or_router, ModelRouter):
        oai = client_or_router._get_openai()
    elif is_anthropic_client(client_or_router):
        print("(OpenAI model listing unavailable — client is Anthropic-only)")
        return
    else:
        oai = client_or_router
    print("Fetching OpenAI-compatible models...\n")
    try:
        models = oai.models.list()
        model_list = sorted(models.data, key=lambda item: item.id)
        print(f"Found {len(model_list)} models:\n")
        for m in model_list:
            print(f"  - {m.id}")
        print()
    except Exception as exc:
        print(f"Failed to list models: {exc}")

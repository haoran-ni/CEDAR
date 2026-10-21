"""Provider-neutral access to language models, with one adapter per provider."""

from sim2real.models.config import ModelConfig, load_model_config, write_starter
from sim2real.models.credentials import (
    CredentialStore,
    InsecureCredentials,
    MissingCredential,
    redact,
    scrubbed_environment,
)
from sim2real.models.types import (
    Message,
    ModelClient,
    ModelError,
    ModelReply,
    ModelSettings,
    TextPart,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

ADAPTERS = ("anthropic",)


def create_client(settings: ModelSettings, credentials: CredentialStore) -> ModelClient:
    """The adapter for ``settings.provider``, holding that provider's key."""
    if settings.provider == "anthropic":
        from sim2real.models.anthropic_client import AnthropicClient

        return AnthropicClient(credentials.get("anthropic"))
    raise ValueError(f"no adapter for provider {settings.provider!r}; available: {', '.join(ADAPTERS)}")


__all__ = [
    "ADAPTERS", "CredentialStore", "InsecureCredentials", "Message", "MissingCredential", "ModelClient",
    "ModelConfig", "ModelError", "ModelReply", "ModelSettings", "TextPart", "ToolCall", "ToolResult", "ToolSpec",
    "Usage", "create_client", "load_model_config", "redact", "scrubbed_environment", "write_starter",
]

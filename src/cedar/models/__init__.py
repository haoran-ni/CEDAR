"""Provider-neutral access to language models, with one adapter per provider."""

from cedar.models.config import ModelConfig, load_model_config, write_starter
from cedar.models.credentials import (
    CredentialStore,
    InsecureCredentials,
    MissingCredential,
    redact,
    scrubbed_environment,
)
from cedar.models.types import (
    MediaPart,
    Message,
    ModelClient,
    ModelError,
    ModelPrices,
    ModelReply,
    ModelSettings,
    TextPart,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    WebAccess,
)

ADAPTERS = ("anthropic",)


def create_client(settings: ModelSettings, credentials: CredentialStore) -> ModelClient:
    """The adapter for ``settings.provider``, holding that provider's key."""
    if settings.provider == "anthropic":
        from cedar.models.anthropic_client import AnthropicClient

        return AnthropicClient(credentials.get("anthropic"))
    raise ValueError(f"no adapter for provider {settings.provider!r}; available: {', '.join(ADAPTERS)}")


__all__ = [
    "ADAPTERS", "CredentialStore", "InsecureCredentials", "MediaPart", "Message", "MissingCredential", "ModelClient",
    "ModelConfig", "ModelError", "ModelPrices", "ModelReply", "ModelSettings", "TextPart", "ToolCall", "ToolResult", "ToolSpec",
    "Usage", "WebAccess", "create_client", "load_model_config", "redact", "scrubbed_environment", "write_starter",
]

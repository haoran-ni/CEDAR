"""Keeping credentials out of everything sim2real stores or starts."""

import os
import re
from collections.abc import Iterable, Mapping

PROVIDER_ENVIRONMENT_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}
"""Standard environment variable per provider."""

_SECRET_NAME = re.compile(r"(_API_KEY|_AUTH_TOKEN|_ACCESS_TOKEN)$|SECRET|PASSWORD")
_KEY_SHAPES = re.compile(r"\b(sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,})")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove known keys, and anything shaped like a provider key, from text about to be stored."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return _KEY_SHAPES.sub("[REDACTED]", text)


def scrubbed_environment(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The environment without credentials, for subprocesses such as job submission."""
    known = set(PROVIDER_ENVIRONMENT_VARIABLES.values())
    return {name: value for name, value in environ.items()
            if name not in known and not _SECRET_NAME.search(name)}

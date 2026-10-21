"""API keys: kept in a user-only file, overridable by environment variables, never persisted elsewhere.

Keys are read only by the process that calls the model. They never go into prompts,
the research record, artifacts, logs, or the environment of scheduler jobs (see
``scrubbed_environment``).
"""

import os
import re
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path

from cedar.secrets import PROVIDER_ENVIRONMENT_VARIABLES, redact, scrubbed_environment

ENVIRONMENT_VARIABLES = PROVIDER_ENVIRONMENT_VARIABLES

_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class MissingCredential(RuntimeError):
    pass


class InsecureCredentials(RuntimeError):
    pass


class CredentialStore:
    def __init__(self, path: str | Path, environ: Mapping[str, str] = os.environ) -> None:
        self.path = Path(path)
        self._environ = environ

    def get(self, provider: str) -> str:
        variable = ENVIRONMENT_VARIABLES.get(provider)
        if variable and self._environ.get(variable):
            return self._environ[variable]
        key = self._read().get(provider)
        if not key:
            hint = f" or set {variable}" if variable else ""
            raise MissingCredential(f"no API key for {provider}; run `cedar auth add {provider}`{hint}")
        return key

    def status(self) -> dict[str, str]:
        """Where each provider's key comes from ("environment" or "file"); never the key itself."""
        sources = {provider: "file" for provider in self._read()}
        for provider, variable in ENVIRONMENT_VARIABLES.items():
            if self._environ.get(variable):
                sources[provider] = "environment"
        return dict(sorted(sources.items()))

    def set(self, provider: str, key: str) -> None:
        _check_provider(provider)
        key = key.strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("an API key cannot be empty or contain whitespace")
        keys = self._read()
        keys[provider] = key
        self._write(keys)

    def remove(self, provider: str) -> bool:
        keys = self._read()
        if keys.pop(provider, None) is None:
            return False
        self._write(keys)
        return True

    def secrets(self) -> list[str]:
        """Every key this process could use, for redaction."""
        found = list(self._read().values())
        found += [self._environ[v] for v in ENVIRONMENT_VARIABLES.values() if self._environ.get(v)]
        return found

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        mode = self.path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise InsecureCredentials(
                f"{self.path} is accessible to other users (mode {stat.filemode(mode)}); run `chmod 600 {self.path}`")
        with self.path.open("rb") as stream:
            data = tomllib.load(stream)
        return {provider: str(entry["api_key"]) for provider, entry in data.items()
                if isinstance(entry, dict) and entry.get("api_key")}

    def _write(self, keys: dict[str, str]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lines = ["# CEDAR API keys. Keep this file private (mode 600); never commit it."]
        for provider, key in sorted(keys.items()):
            _check_provider(provider)
            lines += ["", f"[{provider}]", f"api_key = {_toml_string(key)}"]
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("\n".join(lines) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


def _check_provider(provider: str) -> None:
    if not _PROVIDER.match(provider):
        raise ValueError(f"not a provider name: {provider!r}")


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


__all__ = ["CredentialStore", "ENVIRONMENT_VARIABLES", "InsecureCredentials", "MissingCredential", "redact",
           "scrubbed_environment"]

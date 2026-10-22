"""Which model each agent role uses, from the user's ``models.toml``.

Roles fall back to ``default``. Model names and providers are configuration: the starter
file written by ``cedar models init`` is only a suggestion the user edits.
"""

import tomllib
from pathlib import Path

from cedar.contracts.base import Contract
from cedar.models.types import ModelSettings

STARTER = """# Which model each CEDAR agent role uses. Roles without a section use [roles.default].
# provider: an adapter CEDAR knows ("anthropic"); model: the provider's exact model id.
# Budgets are in dollars. CEDAR knows the published prices of the current Claude models; for any other model, or
# when prices change, give them in dollars per million tokens (web_search: dollars per search), e.g.
#   [roles.default.prices]
#   input = 4.0
#   output = 20.0
#   cache_read = 0.20
#   cache_write = 5.0
#   web_search = 0.01

[roles.default]
provider = "anthropic"
model = "claude-opus-5-5"
effort = "high"
"""


class ModelConfig(Contract):
    roles: dict[str, ModelSettings]

    def for_role(self, role: str) -> ModelSettings:
        settings = self.roles.get(role) or self.roles.get("default")
        if settings is None:
            raise KeyError(f"models.toml has no [roles.{role}] or [roles.default] section")
        return settings


def load_model_config(path: str | Path) -> ModelConfig:
    with open(path, "rb") as stream:
        return ModelConfig.model_validate(tomllib.load(stream))


def write_starter(path: str | Path) -> bool:
    """Write the starter configuration unless a file is already there."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(STARTER)
    return True

"""Where sim2real keeps per-user configuration."""

import os
from collections.abc import Mapping
from pathlib import Path


def config_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """``$SIM2REAL_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/sim2real``, else ``~/.config/sim2real``."""
    if environ.get("SIM2REAL_CONFIG_DIR"):
        return Path(environ["SIM2REAL_CONFIG_DIR"])
    base = environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sim2real"

"""Where sim2real keeps per-user configuration and research data."""

import os
from collections.abc import Mapping
from pathlib import Path


def data_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """Research data (event log, artifacts, transcripts): ``$SIM2REAL_DATA_DIR``, else
    ``$XDG_DATA_HOME/sim2real``, else ``~/.local/share/sim2real``."""
    if environ.get("SIM2REAL_DATA_DIR"):
        return Path(environ["SIM2REAL_DATA_DIR"])
    base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "sim2real"


def config_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """``$SIM2REAL_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/sim2real``, else ``~/.config/sim2real``."""
    if environ.get("SIM2REAL_CONFIG_DIR"):
        return Path(environ["SIM2REAL_CONFIG_DIR"])
    base = environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sim2real"

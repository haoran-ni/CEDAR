"""Where CEDAR keeps per-user configuration and research data."""

import os
from collections.abc import Mapping
from pathlib import Path


def data_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """Research data (event log, artifacts, transcripts): ``$CEDAR_DATA_DIR``, else
    ``$XDG_DATA_HOME/cedar``, else ``~/.local/share/cedar``."""
    if environ.get("CEDAR_DATA_DIR"):
        return Path(environ["CEDAR_DATA_DIR"])
    base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "cedar"


def config_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """``$CEDAR_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/cedar``, else ``~/.config/cedar``."""
    if environ.get("CEDAR_CONFIG_DIR"):
        return Path(environ["CEDAR_CONFIG_DIR"])
    base = environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "cedar"

"""Command-line entry point (``sim2real``).

Only ``--version`` exists so far; subcommands such as ``start`` are added when the
features behind them work.
"""

import argparse

from sim2real import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sim2real",
        description="Sim2Real Scientist (pre-alpha: no investigation commands are implemented yet).",
    )
    parser.add_argument("--version", action="version", version=f"sim2real {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0

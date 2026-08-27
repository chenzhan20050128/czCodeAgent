"""Command-line interface for mca."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the minimal command line while the runtime is scaffolded."""

    parser = argparse.ArgumentParser(
        prog="mca",
        description="Run the mca coding agent.",
    )
    parser.parse_args(argv)
    return 0

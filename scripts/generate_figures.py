"""Run the retained figure and summary-table generation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_COMMANDS: list[list[str]] = [
    ["generate_figures.py"],
    ["summary_tables.py"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Release root directory. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the figure-generation commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for command_tail in DEFAULT_COMMANDS:
        command = [sys.executable, str(root / "experiments" / command_tail[0]), *command_tail[1:]]
        print("command=" + " ".join(command))
        if not args.list_only:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

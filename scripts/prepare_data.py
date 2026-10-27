"""Create the release data directories and print the expected dataset layout."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Release root directory. Defaults to the parent of scripts/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_dir = root / "data"
    cifar10_dir = data_dir / "cifar10"
    data_dir.mkdir(parents=True, exist_ok=True)
    cifar10_dir.mkdir(parents=True, exist_ok=True)

    print(f"release_root={root.as_posix()}")
    print(f"data_dir={data_dir.as_posix()}")
    print(f"cifar10_dir={cifar10_dir.as_posix()}")
    print("status=ready_for_local_copy_or_dataset_download")


if __name__ == "__main__":
    main()

"""Report the expected release checkpoint locations."""

from __future__ import annotations

import argparse
from pathlib import Path


CHECKPOINT_RELATIVE_PATHS = [
    Path("mlp") / "cifar10_mlp.pt",
    Path("resnet") / "cifar10_resnet.pt",
    Path("transformer") / "cifar10_transformer.pt",
    Path("vit") / "cifar10_vit.pt",
]


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
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"release_root={root.as_posix()}")
    print(f"checkpoints_dir={checkpoints_dir.as_posix()}")
    for relative_path in CHECKPOINT_RELATIVE_PATHS:
        path = checkpoints_dir / relative_path
        print(f"{relative_path.as_posix()}={'present' if path.exists() else 'missing'}")


if __name__ == "__main__":
    main()

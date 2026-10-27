"""M1 / Block 2: topology-tampering attacks on ResNet."""

from __future__ import annotations

import argparse
from pathlib import Path

from group_auth_dnn.config import RESNET_INPUT_CHANNELS, RESNET_NUM_CLASSES
from group_auth_dnn.models.resnet import create_resnet_shards

if __package__ in (None, ""):
    from m1_runners import build_cifar10_inputs, run_block2_experiment
else:
    from .m1_runners import build_cifar10_inputs, run_block2_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topologies", type=int, nargs="+", default=[5])
    parser.add_argument("--quant-mode", type=str, default="int8")
    parser.add_argument("--model-version", type=str, default="m1-resnet-v1")
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_samples = build_cifar10_inputs(args.num_samples, args.seed, root=args.data_root)

    summary = run_block2_experiment(
        model_name="ResNet",
        model_slug="resnet",
        num_samples=args.num_samples,
        seed=args.seed,
        topologies=args.topologies,
        quant_mode=args.quant_mode,
        model_version=args.model_version,
        input_samples=input_samples,
        create_shards_fn=lambda *, num_shards, seed: create_resnet_shards(
            RESNET_NUM_CLASSES,
            RESNET_INPUT_CHANNELS,
            num_shards,
            seed=seed,
        ),
        latest_json_path=Path("results") / "m1_block2_resnet.json",
        latest_csv_path=Path("results") / "m1_block2_resnet.csv",
        data_source="CIFAR-10 test subset",
    )

    print("summary_json=results/m1_block2_resnet.json")
    for depth in args.topologies:
        info = summary["topologies"][str(depth)]
        print(
            f"N={depth} "
            f"skip={info['skip_node']['far']:.3f} "
            f"swap={info['swap_order']['far']:.3f} "
            f"replace={info['replace_node']['far']:.3f} "
            f"inject={info['inject_params']['far']:.3f}"
        )


if __name__ == "__main__":
    main()

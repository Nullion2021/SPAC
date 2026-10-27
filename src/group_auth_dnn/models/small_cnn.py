"""Shardable small CNN used by the M5 heterogeneous real-chain runs."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """A compact ReLU CNN that can be split into sequential shards."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: Sequence[int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c3, num_classes),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def split(self, num_shards: int) -> List[nn.Module]:
        if num_shards > len(self.blocks):
            raise ValueError(f"Cannot split {len(self.blocks)} blocks into {num_shards} shards")
        base, remainder = divmod(len(self.blocks), num_shards)
        shards: list[nn.Module] = []
        offset = 0
        for shard_idx in range(num_shards):
            shard_size = base + (1 if shard_idx < remainder else 0)
            shard = nn.Sequential(*list(self.blocks[offset : offset + shard_size]))
            shards.append(shard.eval())
            offset += shard_size
        return shards


def create_small_cnn_shards(
    *,
    num_classes: int = 10,
    in_channels: int = 3,
    channels: Sequence[int] = (16, 32, 64),
    num_shards: int,
    seed: int = 42,
) -> List[nn.Module]:
    torch.manual_seed(seed)
    model = SmallCNN(in_channels=in_channels, num_classes=num_classes, channels=channels)
    return model.split(num_shards)

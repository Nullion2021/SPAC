"""Split ResNet-18 model and shard helpers."""

import torch
import torch.nn as nn
from typing import List


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SplitResNet18(nn.Module):
    """A lightweight ResNet-18 variant that can be split by stage."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        c = [16, 32, 64, 128]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c[0], c[1], num_blocks=2, stride=2)
        self.stage2 = self._make_stage(c[1], c[2], num_blocks=2, stride=2)
        self.stage3 = self._make_stage(c[2], c[3], num_blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c[3], num_classes),
        )
        self.stages = [self.stem, self.stage1, self.stage2, self.stage3, self.head]

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def _flatten_stages(self) -> list:
        """Flatten stages into the smallest shardable submodule list."""
        flat = []
        for stage in self.stages:
            if len(list(stage.children())) > 1:
                flat.extend(stage.children())
            else:
                flat.append(stage)
        return list(flat)

    def split(self, num_shards: int) -> List[nn.Module]:
        """Split the model into num_shards sequential shards."""
        flat = self._flatten_stages()
        if num_shards > len(flat):
            raise ValueError(f"Cannot split {len(flat)} sub-modules into {num_shards} shards")
        base, remainder = divmod(len(flat), num_shards)
        shards = []
        idx = 0
        for i in range(num_shards):
            size = base + (1 if i < remainder else 0)
            shard = nn.Sequential(*flat[idx:idx + size])
            shards.append(shard)
            idx += size
        return shards


def create_resnet_shards(
    num_classes: int,
    in_channels: int,
    num_shards: int,
    seed: int = 42,
) -> List[nn.Module]:
    """Create a ResNet-18 and split it into the requested number of shards."""
    torch.manual_seed(seed)
    model = SplitResNet18(num_classes, in_channels)
    return model.split(num_shards)


def create_resnet_model(
    num_classes: int,
    in_channels: int,
    seed: int = 42,
) -> SplitResNet18:
    torch.manual_seed(seed)
    return SplitResNet18(num_classes, in_channels)

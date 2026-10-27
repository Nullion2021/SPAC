"""MLP backbone used for the M0 sanity pipeline."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, use_relu: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """A shardable feed-forward network with a linear last layer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        blocks = []
        for layer_idx in range(num_layers):
            use_relu = layer_idx < num_layers - 1
            blocks.append(MLPBlock(dims[layer_idx], dims[layer_idx + 1], use_relu=use_relu))
        self.blocks = nn.ModuleList(blocks)

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


def create_mlp_shards(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_shards: int,
    seed: int = 42,
) -> List[nn.Module]:
    torch.manual_seed(seed)
    model = MLP(input_dim, hidden_dim, output_dim, num_layers)
    return model.split(num_shards)


def create_mlp_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    seed: int = 42,
) -> MLP:
    torch.manual_seed(seed)
    return MLP(input_dim, hidden_dim, output_dim, num_layers)

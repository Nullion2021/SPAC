"""Lightweight Transformer model and shard helpers."""

import torch
import torch.nn as nn
from typing import List


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class LightTransformer(nn.Module):
    """A lightweight Transformer encoder + classifier that supports layer-wise splitting."""

    def __init__(self, dim: int, heads: int, num_layers: int, seq_len: int, num_classes: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.cls_token_idx = 0
        self.cls_head = _ClsHeadModule(self.head, self.cls_token_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.cls_head(x)

    def split(self, num_shards: int) -> List[nn.Module]:
        """Split the Transformer into num_shards shards."""
        n_blocks = len(self.blocks)
        if num_shards < 1:
            raise ValueError("num_shards must be positive")

        base, remainder = divmod(n_blocks, num_shards)

        shards = []
        idx = 0
        for i in range(num_shards):
            components = []
            if i == 0:
                components.append(_PosEmbedModule(self.pos_embed))
            size = base + (1 if i < remainder else 0)
            for _ in range(size):
                components.append(self.blocks[idx])
                idx += 1
            if i == num_shards - 1:
                components.append(self.cls_head)
            shards.append(nn.Sequential(*components))
        return shards


class _PosEmbedModule(nn.Module):
    """Wrap pos_embed as its own module."""

    def __init__(self, pos_embed: nn.Parameter):
        super().__init__()
        self.pos_embed = pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class _ClsHeadModule(nn.Module):
    """Select the CLS token before applying the classification head."""

    def __init__(self, head: nn.Module, cls_token_idx: int):
        super().__init__()
        self.head = head
        self.cls_token_idx = cls_token_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, self.cls_token_idx])


def create_transformer_shards(
    dim: int,
    heads: int,
    num_layers: int,
    seq_len: int,
    num_classes: int,
    num_shards: int,
    seed: int = 42,
) -> List[nn.Module]:
    """Create a Transformer and split it into the requested number of shards."""
    torch.manual_seed(seed)
    model = LightTransformer(dim, heads, num_layers, seq_len, num_classes)
    return model.split(num_shards)


def create_transformer_model(
    dim: int,
    heads: int,
    num_layers: int,
    seq_len: int,
    num_classes: int,
    seed: int = 42,
) -> LightTransformer:
    torch.manual_seed(seed)
    return LightTransformer(dim, heads, num_layers, seq_len, num_classes)

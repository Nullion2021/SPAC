"""Split-friendly small ViT for CIFAR-10 experiments."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, *, image_size: int, patch_size: int, in_channels: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class AddClsPosEmbed(nn.Module):
    def __init__(self, *, num_patches: int, dim: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_token, x], dim=1) + self.pos_embed


class ViTBlock(nn.Module):
    def __init__(self, *, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        x = x + self.attn(attn_in, attn_in, attn_in)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHead(nn.Module):
    def __init__(self, *, dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x[:, 0])
        return self.fc(x)


class SplitVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        dim: int,
        heads: int,
        num_layers: int,
        mlp_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim,
        )
        self.token_prep = AddClsPosEmbed(num_patches=num_patches, dim=dim)
        self.blocks = nn.ModuleList(
            [ViTBlock(dim=dim, heads=heads, mlp_dim=mlp_dim) for _ in range(num_layers)]
        )
        self.head = ViTHead(dim=dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.token_prep(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def split(self, num_shards: int) -> List[nn.Module]:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        components: list[nn.Module] = [self.patch_embed, self.token_prep, *self.blocks, self.head]
        base, remainder = divmod(len(components), num_shards)
        shards: list[nn.Module] = []
        idx = 0
        for shard_idx in range(num_shards):
            size = base + (1 if shard_idx < remainder else 0)
            shard_components = components[idx : idx + size]
            shards.append(nn.Sequential(*shard_components))
            idx += size
        return shards


def create_vit_shards(
    image_size: int,
    patch_size: int,
    in_channels: int,
    dim: int,
    heads: int,
    num_layers: int,
    mlp_dim: int,
    num_classes: int,
    num_shards: int,
    seed: int = 42,
) -> List[nn.Module]:
    torch.manual_seed(seed)
    model = SplitVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dim=dim,
        heads=heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )
    return model.split(num_shards)


def create_vit_model(
    image_size: int,
    patch_size: int,
    in_channels: int,
    dim: int,
    heads: int,
    num_layers: int,
    mlp_dim: int,
    num_classes: int,
    seed: int = 42,
) -> SplitVisionTransformer:
    torch.manual_seed(seed)
    return SplitVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dim=dim,
        heads=heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )

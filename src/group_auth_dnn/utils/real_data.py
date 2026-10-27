"""CIFAR-10 loading and input-adaptation helpers."""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from group_auth_dnn.config import (
    CIFAR_IMAGE_SIZE,
    DATASET_DIR,
    REAL_DATA_LOADER_BATCH_SIZE,
    REAL_DATA_NUM_WORKERS,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    TRANSFORMER_PATCH_SIZE,
)


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(CIFAR_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )


def _test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )


def get_cifar10_loaders(
    root: str = DATASET_DIR,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    test_batch_size: int = REAL_DATA_LOADER_BATCH_SIZE,
    num_workers: int = TRAIN_NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=_train_transform(),
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=_test_transform(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def get_cifar10_test_loader(
    root: str = DATASET_DIR,
    batch_size: int = REAL_DATA_LOADER_BATCH_SIZE,
    num_workers: int = REAL_DATA_NUM_WORKERS,
) -> DataLoader:
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=_test_transform(),
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def adapt_images_for_mlp(images: torch.Tensor) -> torch.Tensor:
    return images.view(images.size(0), -1)


def adapt_images_for_transformer(
    images: torch.Tensor,
    patch_size: int = TRANSFORMER_PATCH_SIZE,
) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got {images.shape}")
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size ({height}, {width}) must be divisible by patch_size={patch_size}"
        )
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    num_patches = (height // patch_size) * (width // patch_size)
    patch_dim = channels * patch_size * patch_size
    return patches.view(batch, num_patches, patch_dim)


def adapt_batch_for_model(images: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "mlp":
        return adapt_images_for_mlp(images)
    if model_type in {"resnet", "vit"}:
        return images
    if model_type == "transformer":
        return adapt_images_for_transformer(images)
    raise ValueError(f"Unknown model_type: {model_type}")


def collect_real_test_inputs(
    model_type: str,
    num_samples: int,
    root: str = DATASET_DIR,
    batch_size: int = REAL_DATA_LOADER_BATCH_SIZE,
    num_workers: int = REAL_DATA_NUM_WORKERS,
) -> List[torch.Tensor]:
    loader = get_cifar10_test_loader(root=root, batch_size=batch_size, num_workers=num_workers)
    out: list[torch.Tensor] = []
    for images, _ in loader:
        adapted = adapt_batch_for_model(images, model_type)
        for i in range(adapted.size(0)):
            out.append(adapted[i : i + 1].clone())
            if len(out) >= num_samples:
                return out
    return out

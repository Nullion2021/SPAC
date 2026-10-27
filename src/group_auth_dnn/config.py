"""Global experiment constants and deterministic key generation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Reproducibility
SEED = 42

# Model parameters
MLP_INPUT_DIM = 64
MLP_HIDDEN_DIM = 128
MLP_OUTPUT_DIM = 10
MLP_NUM_LAYERS = 6

RESNET_NUM_CLASSES = 10
RESNET_INPUT_CHANNELS = 3
RESNET_INPUT_SIZE = 32

SMALL_CNN_NUM_CLASSES = 10
SMALL_CNN_INPUT_CHANNELS = 3
SMALL_CNN_CHANNELS = [16, 32, 64]

TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 6
TRANSFORMER_SEQ_LEN = 16
TRANSFORMER_NUM_CLASSES = 10
# Legacy CIFAR-10 patchification helper: 32x32 images -> 16 tokens => patch size 8.
TRANSFORMER_PATCH_SIZE = 8

VIT_IMAGE_SIZE = 32
VIT_PATCH_SIZE = 4
VIT_DIM = 64
VIT_HEADS = 4
VIT_LAYERS = 6
VIT_MLP_DIM = 256
VIT_NUM_CLASSES = 10

# Distributed topology depths
TOPOLOGY_DEPTHS = [3, 5, 10]

# Quantization presets
QUANT_CONFIGS = {
    "fp32": {"mode": "fp32"},
    "fp16": {"mode": "fp16"},
    "int8": {"mode": "int8", "num_bits": 8},
}

EXP3_QUANT_MODES = ["fp16", "int8", "mixed"]

# Task tolerance and protocol setup
DELTA_TASK = 1e-3
HMAC_KEY_LEN = 32

# Release-root paths
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
RELEASE_ROOT = SRC_DIR.parent

# Experiment setup
EXP_NUM_SAMPLES = 200
RESULTS_DIR = str(RELEASE_ROOT / "results")

# Dataset / checkpoint paths used by the reproducibility helpers.
DATASET_DIR = str(RELEASE_ROOT / "data" / "cifar10")
CHECKPOINTS_DIR = str(RELEASE_ROOT / "checkpoints")

# Historical real-data helpers expect CIFAR-10-specific knobs.
CIFAR_IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 0
REAL_DATA_LOADER_BATCH_SIZE = 128
REAL_DATA_NUM_WORKERS = 0

# The current workspace does not ship all of these checkpoints, but we keep a
# stable mapping so evaluation scripts can report missing artifacts explicitly.
CHECKPOINT_PATHS = {
    "mlp": os.path.join(CHECKPOINTS_DIR, "mlp", "cifar10_mlp.pt"),
    "resnet": os.path.join(CHECKPOINTS_DIR, "resnet", "cifar10_resnet.pt"),
    "transformer": os.path.join(CHECKPOINTS_DIR, "transformer", "cifar10_transformer.pt"),
    "vit": os.path.join(CHECKPOINTS_DIR, "vit", "cifar10_vit.pt"),
}

# Optional training presets referenced by historical bytecode-only utilities.
TRAIN_MLP = {
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "target_acc": 0.0,
}
TRAIN_RESNET = {
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "target_acc": 0.0,
}
TRAIN_TRANSFORMER = {
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "target_acc": 0.0,
}

# Directory and threshold placeholders for real-data validation helpers.
REAL_VALIDATION_DIR = os.path.join(RESULTS_DIR, "real_validation")
REAL_VALIDATION_THRESHOLDS = {}


def generate_key(seed: int = None) -> bytes:
    """Generate a deterministic HMAC key when a seed is provided."""
    if seed is None:
        return os.urandom(HMAC_KEY_LEN)
    return hashlib.sha256(seed.to_bytes(8, "big")).digest()[:HMAC_KEY_LEN]

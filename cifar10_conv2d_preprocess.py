#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 Conv2D/ResNet Preprocess (PyTorch / torchvision backend)

- No flatten, keep [N, 32, 32, 3]
- Stratified train/val split
- Per-channel (3-dim) normalization: (x - mean[c]) / std[c]
- Save: split.json, norm.json, meta.json
- (Optional) export normalized arrays for Conv2D
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from data_cifar10 import load_cifar10


def as_python(obj):
    """Convert numpy types to pure Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def stratified_split(y, seed: int, val_frac: float = 0.1):
    """
    Perform a stratified train/val split on labels y.

    Args:
        y: 1D array of labels, shape [N]
        seed: random seed for the split
        val_frac: fraction of samples used for validation

    Returns:
        idx_train, idx_val: index arrays for train and validation subsets
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    idx_train, idx_val = next(sss.split(np.zeros_like(y), y))
    return idx_train, idx_val


def compute_channel_norm_stats(x_subset: np.ndarray, eps: float = 1e-6):
    """
    Compute per-channel mean and std.

    Args:
        x_subset: float32 array of shape [N, 32, 32, 3], typically train subset
        eps: small epsilon for numerical stability

    Returns:
        mean: float32 array of shape [3]
        std:  float32 array of shape [3]
    """
    mean = x_subset.mean(axis=(0, 1, 2))  # per-channel
    var = x_subset.var(axis=(0, 1, 2))
    std = np.sqrt(np.maximum(var, eps))
    return mean, std


def apply_channel_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Apply per-channel standardization to images.

    Args:
        x: float32 array of shape [N, 32, 32, 3]
        mean: float32 array of shape [3]
        std:  float32 array of shape [3]

    Returns:
        x_norm: float32 array of the same shape as x
    """
    return (x - mean[None, None, None, :]) / std[None, None, None, :]


def save_json(obj, path: Path):
    """
    Save a Python object as JSON to the given path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CIFAR-10 for Conv2D/ResNet")
    parser.add_argument("--seed", type=int, default=42,
                        help="data split seed (used for train/val split)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="fraction of training data used for validation")
    parser.add_argument("--artifacts_dir", type=str,
                        default="artifacts/cifar10_conv2d_preprocess",
                        help="directory to store preprocess artifacts")
    parser.add_argument("--export_npz", action="store_true",
                        help="also export normalized numpy arrays as npz")
    args = parser.parse_args()

    seed = args.seed
    out_dir = Path(args.artifacts_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Load CIFAR-10 via torchvision (through data_cifar10) ----------------
    # x_*: uint8 [N, 32, 32, 3] in [0, 255]
    # y_*: int64 [N]
    (x_train_all, y_train_all), (x_test, y_test) = load_cifar10()

    # Cast to float32 in [0, 1]
    x_train_all = x_train_all.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # ---------------- 2. Train/Val split ----------------
    idx_train, idx_val = stratified_split(y_train_all, seed, val_frac=args.val_frac)

    x_tr, y_tr = x_train_all[idx_train], y_train_all[idx_train]
    x_val, y_val = x_train_all[idx_val], y_train_all[idx_val]

    # ---------------- 3. Compute per-channel mean/std on train subset ----------------
    mean, std = compute_channel_norm_stats(x_tr)

    # ---------------- 4. Save split.json ----------------
    split_json = {
        "seed": seed,
        "val_frac": args.val_frac,
        "train_idx": as_python(idx_train),
        "val_idx": as_python(idx_val),
        "counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(x_test)),
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(split_json, out_dir / "split.json")

    # ---------------- 5. Save norm.json ----------------
    norm_json = {
        "type": "per_channel_standardization",
        "channels": 3,
        "mean": as_python(mean),
        "std": as_python(std),
        "epsilon": 1e-6,
        "fitted_on": "train_only",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(norm_json, out_dir / "norm.json")

    # ---------------- 6. Save meta.json ----------------
    meta_json = {
        "dataset": "CIFAR-10",
        "train_shape": list(x_train_all.shape),
        "test_shape": list(x_test.shape),
        "dtype": "float32",
        "y_train_minmax": [int(np.min(y_train_all)), int(np.max(y_train_all))],
        "y_test_minmax": [int(np.min(y_test)), int(np.max(y_test))],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(meta_json, out_dir / "meta.json")

    print(f"[OK] Wrote artifacts to: {out_dir.resolve()}")
    print(" - split.json")
    print(" - norm.json")
    print(" - meta.json")

    # ---------------- 7. Optional: export normalized numpy arrays ----------------
    if args.export_npz:
        x_tr_norm = apply_channel_norm(x_tr, mean, std)
        x_val_norm = apply_channel_norm(x_val, mean, std)
        x_test_norm = apply_channel_norm(x_test, mean, std)

        npz_path = out_dir / f"cifar10_conv2d_seed{seed}_normed.npz"
        np.savez_compressed(
            npz_path,
            x_train=x_tr_norm,
            y_train=y_tr,
            x_val=x_val_norm,
            y_val=y_val,
            x_test=x_test_norm,
            y_test=y_test,
            mean=mean,
            std=std,
        )
        print(f"[OK] Exported normalized Conv2D arrays to: {npz_path.resolve()}")


if __name__ == "__main__":
    main()

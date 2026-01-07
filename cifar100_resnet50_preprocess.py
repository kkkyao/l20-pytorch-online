#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-100 preprocess for ResNet-50 training (Strategy B only).

Offline artifacts:
  - split.json: stratified train/val indices
  - norm.json : per-channel mean/std computed on TRAIN SUBSET ONLY
  - meta.json : dataset metadata

Notes:
  - No export of normalized arrays (npz). ResNet pipelines should normalize online
    in torchvision transforms for flexibility.
  - Mean/std is computed ONLY from the stratified train subset (not val/test).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from data_cifar100 import load_cifar100


def as_python(obj):
    """Convert numpy types to pure Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def stratified_split(y: np.ndarray, seed: int, val_frac: float):
    """Stratified train/val split. Returns idx_train, idx_val."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_frac), random_state=int(seed))
    idx_train, idx_val = next(sss.split(np.zeros_like(y), y))
    return idx_train, idx_val


def compute_channel_norm_stats(x_subset: np.ndarray, eps: float = 1e-6):
    """
    Compute per-channel mean/std.

    Args:
        x_subset: float32 array [N, 32, 32, 3] in [0, 1] (TRAIN SUBSET ONLY)
    Returns:
        mean, std: float32 arrays of shape [3]
    """
    mean = x_subset.mean(axis=(0, 1, 2))
    var = x_subset.var(axis=(0, 1, 2))
    std = np.sqrt(np.maximum(var, eps))
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CIFAR-100 for ResNet-50 (train-subset mean/std only)")
    # Defaults chosen so you can run: python cifar100_resnet50_preprocess.py
    parser.add_argument("--seed", type=int, default=42,
                        help="data split seed (used for stratified train/val split)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="fraction of training data used for validation")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts/cifar100_resnet50_preprocess",
                        help="directory to store preprocess artifacts")
    args = parser.parse_args()

    seed = int(args.seed)
    val_frac = float(args.val_frac)

    out_dir = Path(args.artifacts_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CIFAR-100 (numpy)
    # x_*: uint8 [N, 32, 32, 3] in [0, 255]
    # y_*: int64 [N] in [0, 99]
    (x_train_all, y_train_all), (x_test, y_test) = load_cifar100()

    # 2) Stratified split on training set
    idx_train, idx_val = stratified_split(y_train_all, seed, val_frac=val_frac)

    # 3) Compute mean/std on TRAIN SUBSET ONLY
    x_tr = x_train_all[idx_train].astype(np.float32) / 255.0  # float32 in [0,1]
    mean, std = compute_channel_norm_stats(x_tr, eps=1e-6)

    # 4) split.json
    split_json = {
        "seed": seed,
        "val_frac": val_frac,
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

    # 5) norm.json (Strategy B only)
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

    # 6) meta.json
    meta_json = {
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "image_shape_hwc": [32, 32, 3],
        "train_shape": list(x_train_all.shape),
        "test_shape": list(x_test.shape),
        "x_dtype": str(x_train_all.dtype),
        "y_dtype": str(y_train_all.dtype),
        "y_train_minmax": [int(np.min(y_train_all)), int(np.max(y_train_all))],
        "y_test_minmax": [int(np.min(y_test)), int(np.max(y_test))],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(meta_json, out_dir / "meta.json")

    print(f"[OK] Wrote artifacts to: {out_dir.resolve()}")
    print(" - split.json")
    print(" - norm.json (mean/std fitted on TRAIN SUBSET ONLY)")
    print(" - meta.json")
    print(f"[INFO] mean={mean.tolist()} std={std.tolist()}")


if __name__ == "__main__":
    main()

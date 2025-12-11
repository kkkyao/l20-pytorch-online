#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST-1D MLP preprocessing (PyTorch-friendly).

This script:
  - Creates a fixed stratified train/val split with a given seed.
  - Fits per-feature (length=40) normalization statistics on TRAIN ONLY.
  - Saves artifacts: split.json, norm.json, meta.json.
  - Optionally exports normalized arrays for MLP: shape [N, 40].

The exported artifacts are designed to be consumed by MLP baselines
that take 40-dimensional input vectors (e.g., PyTorch MLP baselines).
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Project-specific loader.
# Expected return: ((x_train, y_train), (x_test, y_test))
from data_mnist1d import load_mnist1d


# ----------------------------- Helper utilities -----------------------------
def as_python(obj):
    """Convert NumPy scalar/array types to native Python types for JSON."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def sha256_of_array(a: np.ndarray, max_bytes: int = 8_000_000) -> str:
    """
    Compute a SHA256 digest of the array contents.

    For very large arrays, only the first `max_bytes` bytes of the raw buffer
    are used as a fingerprint to avoid excessive memory.
    """
    h = hashlib.sha256()
    b = a.tobytes()
    h.update(b[:max_bytes])
    return h.hexdigest()


def stratified_split(x_train, y_train, seed: int, val_frac: float = 0.1):
    """Return stratified train/val indices from the original training set."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    idx_train, idx_val = next(sss.split(x_train, y_train))
    return idx_train, idx_val


def compute_norm_stats(x_train_subset: np.ndarray, eps: float = 1e-6):
    """
    Compute per-feature mean and std on the TRAIN subset.

    x_train_subset: array of shape [N_train, 40]
    Returns:
      mean: [40]
      std:  [40] (non-zero due to epsilon)
    """
    mean = x_train_subset.mean(axis=0)          # [40]
    var = x_train_subset.var(axis=0, ddof=0)    # [40]
    std = np.sqrt(np.maximum(var, eps))         # ensure non-zero std
    return mean, std


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Apply per-feature standardization using precomputed mean/std.

    Input / output shape: [N, 40]
    """
    return (x - mean[None, :]) / std[None, :]


def to_N40(x: np.ndarray) -> np.ndarray:
    """
    Normalize the shape of x to [N, 40].

    Supported input shapes:
      - [N, 40]
      - [N, 40, 1]
      - [N, 1, 40]
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 40:
        return x
    if x.ndim == 3:
        if x.shape[1] == 40 and x.shape[2] == 1:   # [N,40,1] -> [N,40]
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == 40:   # [N,1,40] -> [N,40]
            return x[:, 0, :]
    raise AssertionError(
        f"Unexpected shape for x: {x.shape}, "
        f"expect [N,40] or [N,40,1]/[N,1,40]"
    )


def save_json(obj: dict, path: Path):
    """Save a Python dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------- Main preprocessing -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess MNIST-1D for MLP.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for stratified split and data loading",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="validation fraction from the training set",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts/preprocess",
        help="directory where artifacts are written",
    )
    parser.add_argument(
        "--export_npz",
        action="store_true",
        help="also export normalized arrays for MLP",
    )
    parser.add_argument(
        "--npz_name",
        type=str,
        default=None,
        help="custom name for exported npz (optional)",
    )
    args = parser.parse_args()

    seed = args.seed
    val_frac = args.val_frac
    out_dir = Path(args.artifacts_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data and normalize shapes/dtypes
    (x_tr, y_tr), (x_te, y_te) = load_mnist1d(length=40, seed=seed)

    x_tr = to_N40(np.asarray(x_tr)).astype(np.float32, copy=False)
    x_te = to_N40(np.asarray(x_te)).astype(np.float32, copy=False)
    y_tr = np.asarray(y_tr).astype(np.int64, copy=False)
    y_te = np.asarray(y_te).astype(np.int64, copy=False)

    assert x_tr.ndim == 2 and x_tr.shape[1] == 40, \
        f"Expected x_train shape [N,40], got {x_tr.shape}"
    assert x_te.ndim == 2 and x_te.shape[1] == 40, \
        f"Expected x_test shape [N,40], got {x_te.shape}"

    # 2) Stratified train/val split from the original training set
    idx_train, idx_val = stratified_split(x_tr, y_tr, seed=seed, val_frac=val_frac)
    x_tr_split, y_tr_split = x_tr[idx_train], y_tr[idx_train]
    x_val_split, y_val_split = x_tr[idx_val], y_tr[idx_val]

    # 3) Fit normalization statistics on TRAIN subset only
    mean, std = compute_norm_stats(x_tr_split, eps=1e-6)

    # 4) Save split.json
    split_json = {
        "seed": seed,
        "val_frac": val_frac,
        "train_idx": as_python(idx_train),
        "val_idx": as_python(idx_val),
        "counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(x_te)),
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(split_json, out_dir / "split.json")

    # 5) Save norm.json (only statistics, no raw data)
    norm_json = {
        "type": "per_feature_standardization",
        "feature_length": 40,
        "mean": as_python(mean),
        "std": as_python(std),
        "epsilon": 1e-6,
        "fitted_on": "train_only",
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(norm_json, out_dir / "norm.json")

    # 6) Save meta.json (dataset fingerprints, label ranges, etc.)
    meta_json = {
        "dataset": "MNIST-1D",
        "train_shape": list(x_tr.shape),
        "test_shape": list(x_te.shape),
        "dtype": str(x_tr.dtype),
        "train_sha256": sha256_of_array(x_tr),
        "test_sha256": sha256_of_array(x_te),
        "y_train_minmax": [int(np.min(y_tr)), int(np.max(y_tr))],
        "y_test_minmax": [int(np.min(y_te)), int(np.max(y_te))],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_json(meta_json, out_dir / "meta.json")

    print(f"[OK] Wrote artifacts to: {out_dir.resolve()}")
    print(" - split.json")
    print(" - norm.json")
    print(" - meta.json")

    # 7) Optional: export normalized arrays for MLP
    if args.export_npz:
        x_train_norm = apply_normalization(x_tr_split, mean, std)   # [N_train, 40]
        x_val_norm = apply_normalization(x_val_split, mean, std)    # [N_val, 40]
        x_test_norm = apply_normalization(x_te, mean, std)          # [N_test, 40]

        npz_name = args.npz_name or f"mnist1d_seed{seed}_mlp_normed.npz"
        npz_path = out_dir / npz_name
        np.savez_compressed(
            npz_path,
            x_train=x_train_norm.astype(np.float32, copy=False),
            y_train=y_tr_split,
            x_val=x_val_norm.astype(np.float32, copy=False),
            y_val=y_val_split,
            x_test=x_test_norm.astype(np.float32, copy=False),
            y_test=y_te,
            mean=mean,
            std=std,
            seed=seed,
            shape="N40",
        )
        print(f"[OK] Exported normalized MLP arrays to: {npz_path.resolve()}")
        print(
            f"    x_train: {x_train_norm.shape}, "
            f"x_val: {x_val_norm.shape}, x_test: {x_test_norm.shape}"
        )


if __name__ == "__main__":
    main()

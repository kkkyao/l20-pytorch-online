#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for loading CIFAR-10 as numpy arrays.

This module provides a single function:

    load_cifar10(data_dir="data", cache_file="cifar10_numpy.npz")

which returns:
    ( (x_train, y_train), (x_test, y_test) )

where:
    - x_* are uint8 images in [0, 255] with shape [N, 32, 32, 3]
    - y_* are int64 labels in [0, 9] with shape [N]
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torchvision
from torchvision.datasets import CIFAR10


def _build_numpy_from_torchvision(dataset: CIFAR10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a torchvision CIFAR10 dataset (with PIL images) into numpy arrays.

    Returns:
        x: uint8 array of shape [N, 32, 32, 3]
        y: int64 array of shape [N]
    """
    images = []
    labels = []
    for img, label in dataset:
        # img is a PIL Image in mode "RGB", size (32, 32)
        arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
        images.append(arr)
        labels.append(label)

    x = np.stack(images, axis=0)  # [N, 32, 32, 3]
    y = np.array(labels, dtype=np.int64)
    return x, y


def load_cifar10(
    data_dir: str = "data",
    cache_file: str = "cifar10_numpy.npz",
):
    """
    Load CIFAR-10 as numpy arrays, downloading if needed.

    This function will:
      1. Check if a cached npz file exists under `data_dir/cache_file`.
      2. If it exists, load arrays from the cache.
      3. Otherwise, download CIFAR-10 using torchvision, convert to numpy,
         save to cache, and then return the arrays.

    Args:
        data_dir: Directory where CIFAR-10 will be downloaded / cached.
        cache_file: Name of the npz file used for caching numpy arrays.

    Returns:
        ( (x_train, y_train), (x_test, y_test) ), where:
            x_train: uint8 array [N_train, 32, 32, 3]
            y_train: int64 array [N_train]
            x_test:  uint8 array [N_test, 32, 32, 3]
            y_test:  int64 array [N_test]
    """
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    cache_path = data_dir_path / cache_file

    if cache_path.exists():
        # Load from cached npz
        npz = np.load(cache_path)
        x_train = npz["x_train"]
        y_train = npz["y_train"]
        x_test = npz["x_test"]
        y_test = npz["y_test"]
        return (x_train, y_train), (x_test, y_test)

    # Cache does not exist: download via torchvision and build numpy arrays
    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir_path),
        train=True,
        download=True,
        transform=None,  # keep PIL Images, we will convert manually
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir_path),
        train=False,
        download=True,
        transform=None,
    )

    x_train, y_train = _build_numpy_from_torchvision(train_set)
    x_test, y_test = _build_numpy_from_torchvision(test_set)

    # Save cache for future calls
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # Simple sanity check
    (x_tr, y_tr), (x_te, y_te) = load_cifar10()
    print("x_train:", x_tr.shape, x_tr.dtype, "[min,max] =", x_tr.min(), x_tr.max())
    print("y_train:", y_tr.shape, y_tr.dtype)
    print("x_test:", x_te.shape, x_te.dtype, "[min,max] =", x_te.min(), x_te.max())
    print("y_test:", y_te.shape, y_te.dtype)

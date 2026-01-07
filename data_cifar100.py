#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for loading CIFAR-100 as numpy arrays.

Returns:
    ( (x_train, y_train), (x_test, y_test) )

where:
    - x_* are uint8 images in [0, 255] with shape [N, 32, 32, 3]
    - y_* are int64 labels in [0, 99] with shape [N]
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torchvision
from torchvision.datasets import CIFAR100


def _build_numpy_from_torchvision(dataset: CIFAR100) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for img, label in dataset:
        arr = np.array(img, dtype=np.uint8)
        images.append(arr)
        labels.append(label)
    x = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    return x, y


def load_cifar100(
    data_dir: str = "data",
    cache_file: str = "cifar100_numpy.npz",
):
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir_path / cache_file

    if cache_path.exists():
        npz = np.load(cache_path)
        return (npz["x_train"], npz["y_train"]), (npz["x_test"], npz["y_test"])

    train_set = torchvision.datasets.CIFAR100(
        root=str(data_dir_path),
        train=True,
        download=True,
        transform=None,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(data_dir_path),
        train=False,
        download=True,
        transform=None,
    )

    x_train, y_train = _build_numpy_from_torchvision(train_set)
    x_test, y_test = _build_numpy_from_torchvision(test_set)

    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    return (x_train, y_train), (x_test, y_test)

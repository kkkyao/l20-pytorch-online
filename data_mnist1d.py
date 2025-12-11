#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for loading the MNIST-1D dataset.

This module provides a single public function:

    load_mnist1d(length=40, seed=0, root="artifacts/mnist1d", download=True)

which returns:
    ( (x_train, y_train), (x_test, y_test) )

The dataset is the official MNIST-1D release from:
    https://github.com/greydanus/mnist1d

Implementation details:
  * On first call, the code downloads `mnist1d_data.pkl` from GitHub
    and caches it under `root` (default: artifacts/mnist1d/).
  * On later calls, it only reads the cached file.
  * `length` is expected to be 40 (the default MNIST-1D sequence length).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from urllib.request import urlopen, URLError

MNIST1D_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"


def _download_mnist1d(root: Path) -> Path:
    """
    Download the frozen MNIST-1D dataset (mnist1d_data.pkl) to `root`.

    If the file already exists, it is left untouched and its path is returned.
    """
    root.mkdir(parents=True, exist_ok=True)
    pkl_path = root / "mnist1d_data.pkl"

    if pkl_path.exists():
        return pkl_path

    print(f"[MNIST-1D] Downloading dataset from {MNIST1D_URL} ...")
    try:
        with urlopen(MNIST1D_URL) as resp:
            data_bytes = resp.read()
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download MNIST-1D from {MNIST1D_URL}. "
            f"Please check your Internet connection or download manually."
        ) from exc

    with open(pkl_path, "wb") as f:
        f.write(data_bytes)

    print(f"[MNIST-1D] Saved to: {pkl_path.resolve()}")
    return pkl_path


def load_mnist1d(
    length: int = 40,
    seed: int = 0,
    root: str | Path = "artifacts/mnist1d",
    download: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST-1D as ( (x_train, y_train), (x_test, y_test) ).

    Args
    ----
    length: expected sequence length (default 40). The official dataset
            uses length 40; if this does not match, an error is raised.
    seed:   kept for API compatibility with existing code. The dataset
            itself is deterministic; `seed` is not used here.
    root:   directory where mnist1d_data.pkl will be stored.
    download: if True, download the dataset when it is missing. If False
              and the file is missing, a FileNotFoundError is raised.

    Returns
    -------
    ( (x_train, y_train), (x_test, y_test) )
      x_*: float32 arrays of shape [N, length]
      y_*: int64 arrays of shape [N]
    """
    root = Path(root)
    pkl_path = root / "mnist1d_data.pkl"

    if not pkl_path.exists():
        if not download:
            raise FileNotFoundError(
                f"MNIST-1D pickle not found at {pkl_path}. "
                f"Set download=True or download the file manually."
            )
        pkl_path = _download_mnist1d(root)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Keys in the official file:
    #   'x', 'x_test', 'y', 'y_test', 't', 'templates'
    x_train = np.asarray(data["x"], dtype=np.float32)
    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_train = np.asarray(data["y"], dtype=np.int64)
    y_test = np.asarray(data["y_test"], dtype=np.int64)

    if x_train.ndim != 2:
        raise ValueError(f"Expected x_train to have shape [N, L], got {x_train.shape}")
    if x_test.ndim != 2:
        raise ValueError(f"Expected x_test to have shape [N, L], got {x_test.shape}")
    if x_train.shape[1] != length or x_test.shape[1] != length:
        raise ValueError(
            f"Expected sequence length {length}, but got "
            f"{x_train.shape[1]} (train) and {x_test.shape[1]} (test)."
        )

    return (x_train, y_train), (x_test, y_test)


__all__ = ["load_mnist1d"]

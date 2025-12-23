# loader_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized DataLoader utilities for reproducible sampling across experiments.

Design goals
------------
1) Keep randomness "clean" and comparable across methods:
   - Use independent RNG streams for train and val loaders (separate torch.Generator).
   - Avoid coupling where one method consumes extra RNG (e.g., F1 val sampling) and
     accidentally changes the train shuffling sequence.

2) Support multi-worker loading without "random drift":
   - If num_workers > 0, seed numpy/random inside each worker deterministically.

3) Keep dataset-agnostic:
   - Works for MNIST1D and CIFAR10 (or any torch.utils.data.Dataset).

Usage
-----
from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader

cfg = LoaderCfg(batch_size=args.bs, num_workers=0)

train_loader, val_loader = make_train_val_loaders(
    train_dataset, val_dataset, cfg,
    seed=args.seed,
    train_shuffle=True,
    val_shuffle=True,   # for F1/F2: val batches sampled online
    train_drop_last=True,
    val_drop_last=True,
)

val_eval_loader = make_eval_loader(val_dataset, batch_size=512)
test_loader = make_eval_loader(test_dataset, batch_size=512)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int) -> None:
    """
    Seed numpy/random inside each DataLoader worker process.

    Note:
      - torch.initial_seed() is already set by PyTorch for each worker
        based on the main process seed/generator.
      - We mirror it into numpy/random to make transforms or custom dataset
        code reproducible when they use numpy.random or Python's random.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class LoaderCfg:
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None  # only used when num_workers > 0


def _make_generator(seed: int, offset: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) + int(offset))
    return g


def make_loader(
    dataset: Dataset,
    cfg: LoaderCfg,
    *,
    shuffle: bool,
    drop_last: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """
    Low-level DataLoader constructor used by all helpers.

    Parameters
    ----------
    dataset:
        torch Dataset
    cfg:
        Loader configuration
    shuffle / drop_last:
        standard DataLoader flags
    generator:
        Optional torch.Generator used by the sampler. If provided, it makes
        shuffle order deterministic for the given seed/offset.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        worker_init_fn=(seed_worker if cfg.num_workers > 0 else None),
        generator=generator,
    )
    # prefetch_factor is only valid when num_workers > 0
    if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    return DataLoader(**kwargs)


def make_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: LoaderCfg,
    *,
    seed: int,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    train_drop_last: bool = True,
    val_drop_last: bool = True,
    train_gen_offset: int = 12345,
    val_gen_offset: int = 54321,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders using separate RNG streams (generators).

    Why separate generators?
      - Prevents "RNG coupling": in methods like F1/F2, val batches may be sampled
        every train step and thus consume extra randomness. If train and val share
        the same RNG stream, this would change train shuffling sequence, adding
        unnecessary variance and harming fairness.

    Typical choices:
      - Baselines: val_shuffle=False (eval-style val each epoch)
      - F1/F2:     val_shuffle=True  (online sampling for meta-loss)
    """
    g_train = _make_generator(seed, train_gen_offset) if train_shuffle else None
    g_val = _make_generator(seed, val_gen_offset) if val_shuffle else None

    train_loader = make_loader(
        train_dataset, cfg,
        shuffle=train_shuffle,
        drop_last=train_drop_last,
        generator=g_train,
    )
    val_loader = make_loader(
        val_dataset, cfg,
        shuffle=val_shuffle,
        drop_last=val_drop_last,
        generator=g_val,
    )
    return train_loader, val_loader


def make_eval_loader(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Convenience helper for evaluation loaders (no shuffle, no drop_last).

    We keep eval deterministic by default:
      - shuffle=False
      - drop_last=False
    """
    cfg = LoaderCfg(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(num_workers > 0),
        prefetch_factor=None,
    )
    return make_loader(dataset, cfg, shuffle=False, drop_last=False, generator=None)

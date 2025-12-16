#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline for MNIST-1D with the SAME optimizee as L2L-GDGD script:
  - Optimizee: 1-hidden-layer MLP (sigmoid, 20 hidden units)
  - Task distribution: fixed dataset by data_seed; task differs by minibatch shuffle order (task_seed)
  - Budget: exactly optim_steps minibatch updates (SGD/Adam/RMSprop)
  - Preprocess: optional per-position normalization from preprocess_dir/seed_{data_seed}/norm.json

This is the fair "same steps" comparator for your learned optimizer runs.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d


# ---------------------- utils ----------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_N40(x: np.ndarray, length: int = 40) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}")


def load_norm_stats(preprocess_dir: Path, data_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    p = preprocess_dir / f"seed_{data_seed}" / "norm.json"
    if not p.exists():
        raise FileNotFoundError(f"norm.json not found: {p}")
    with open(p, "r") as f:
        norm = json.load(f)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean[None, :]) / (std[None, :] + eps)


# ---------------------- task stream ----------------------
class MNIST1DTask:
    """A task = minibatch stream (shuffle order) over fixed dataset."""
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, task_seed: int, drop_last: bool = True):
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))

        gen = torch.Generator()
        gen.manual_seed(int(task_seed))

        self.loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=gen,
        )
        self._it = iter(self.loader)

    def sample(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


# ---------------------- optimizee (same as L2O script) ----------------------
class MLPOptimizee(nn.Module):
    def __init__(self, hidden_dim: int = 20, init_scale: float = 1e-3):
        super().__init__()
        self.fc1 = nn.Linear(40, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float):
        # notebook-like small init; important for sigmoid stability
        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def eval_full(model: nn.Module, x_np: np.ndarray, y_np: np.ndarray, batch_size: int = 512) -> Tuple[float, float]:
    device = get_device()
    model.eval()

    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).to(device)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.size(0))

    return total_loss / max(total, 1), total_correct / max(total, 1)


def run_one_task(
    opt_name: str,
    lr: float,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    task_seed: int,
    optim_steps: int,
    batch_size: int,
    init_scale: float,
    weight_decay: float,
) -> Tuple[float, float, float, float]:
    device = get_device()
    task = MNIST1DTask(xtr, ytr, batch_size=batch_size, task_seed=task_seed, drop_last=True)
    model = MLPOptimizee(hidden_dim=20, init_scale=init_scale).to(device)

    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown opt: {opt_name}")

    model.train()
    for _ in range(optim_steps):
        xb, yb = task.sample()
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

    tr_loss, tr_acc = eval_full(model, xtr, ytr, batch_size=512)
    te_loss, te_acc = eval_full(model, xte, yte, batch_size=512)
    return tr_loss, tr_acc, te_loss, te_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)

    ap.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--optim_steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--init_scale", type=float, default=1e-3)

    ap.add_argument("--preprocess_dir", type=str, default=None)

    # match your L2O "report_tasks" logic: average over several task seeds
    ap.add_argument("--report_tasks", type=int, default=3)
    ap.add_argument("--eval_seed_low", type=int, default=10000)
    ap.add_argument("--eval_seed_high", type=int, default=10999)

    args = ap.parse_args()
    set_seed(args.seed)

    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = to_N40(xtr).astype(np.float32)
    xte = to_N40(xte).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    if args.preprocess_dir is not None:
        mean, std = load_norm_stats(Path(args.preprocess_dir), args.data_seed)
        xtr = apply_norm(xtr, mean, std)
        xte = apply_norm(xte, mean, std)

    rng = np.random.RandomState(args.seed)
    def sample_task_seed():
        return int(rng.randint(args.eval_seed_low, args.eval_seed_high + 1))

    tr_losses, tr_accs, te_losses, te_accs = [], [], [], []
    for _ in range(args.report_tasks):
        s = sample_task_seed()
        tr_loss, tr_acc, te_loss, te_acc = run_one_task(
            opt_name=args.opt,
            lr=args.lr,
            xtr=xtr, ytr=ytr, xte=xte, yte=yte,
            task_seed=s,
            optim_steps=args.optim_steps,
            batch_size=args.bs,
            init_scale=args.init_scale,
            weight_decay=args.weight_decay,
        )
        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        te_losses.append(te_loss); te_accs.append(te_acc)

    print(
        f"[{args.opt.upper()}] steps={args.optim_steps} lr={args.lr} init={args.init_scale} "
        f"report_train_loss={float(np.mean(tr_losses)):.4f} report_train_acc={float(np.mean(tr_accs)):.4f} "
        f"report_test_loss={float(np.mean(te_losses)):.4f} report_test_acc={float(np.mean(te_accs)):.4f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learning to Learn by Gradient Descent by Gradient Descent (L2L-GDGD)
Paper-style coordinate-wise LSTM optimizer, BUT with Conv1D optimizee aligned to your F1/F2/F3.

Key alignment to your F1 code:
  - Optimizee architecture: Conv1D backbone (same as build_model() in F1):
        Conv1d(1->32,k=5,pad=2) + ReLU
        MaxPool1d(2)
        Conv1d(32->64,k=5,pad=2) + ReLU
        AdaptiveAvgPool1d(1)
        FC(64->64) + ReLU
        FC(64->10)
  - Data preprocessing:
        uses {preprocess_dir}/seed_{data_seed}/split.json and norm.json
        normalize per-position, then reshape input to [N, 1, 40]
  - Report metrics on train/val/test with same split.

L2L-GDGD specifics:
  - Coordinate-wise optimizer: each scalar parameter has its own LSTM hidden/cell state
  - Gradient preprocessing (Appendix A) optional
  - Truncated BPTT with unroll steps
  - Meta-gradient clipping optional

Outputs:
  - runs/<run_name>/train_log.csv
  - runs/<run_name>/curve_eval_mean.csv   (mean eval-task loss curve per meta-epoch)
  - runs/<run_name>/best_opt.pt
  - runs/<run_name>/result.json
"""

import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

# Prefer the "stateless" functional_call (available in torch>=1.13).
try:
    from torch.nn.utils.stateless import functional_call as _functional_call
except Exception:  # pragma: no cover
    _functional_call = None


# ---------------------- device / seed utils ----------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detach_var(v: torch.Tensor) -> torch.Tensor:
    """
    Detach from previous graph and create a fresh leaf tensor with requires_grad=True.
    Critical for truncated BPTT in L2L-GDGD.
    """
    device = get_device()
    out = v.detach().clone().to(device)
    out.requires_grad_(True)
    out.retain_grad()
    return out


# ---------------------- Data helpers ----------------------
def _to_N40(x: np.ndarray, length: int = 40) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        # [N, 40, 1] or [N, 1, 40]
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}")


def load_split_and_norm(preprocess_dir: Path, data_seed: int) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Expect:
      {preprocess_dir}/seed_{data_seed}/split.json  -> {"train_idx": [...], "val_idx": [...]}
      {preprocess_dir}/seed_{data_seed}/norm.json   -> {"mean": [...len40...], "std": [...len40...]}

    Returns: (split, mean, std)
    """
    pdir = preprocess_dir / f"seed_{data_seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split.json not found: {split_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"norm.json not found: {norm_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return split, mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean[None, :]) / (std[None, :] + eps)


def to_N1L(x: np.ndarray, length: int = 40) -> np.ndarray:
    """
    Convert to [N, 1, L] float32.
    """
    x2 = _to_N40(x, length=length).astype(np.float32)
    return x2[:, None, :]


# ---------------------- Task: minibatch stream (task distribution) ----------------------
class MNIST1DTask:
    """
    A task is a minibatch stream over a (subsampled) dataset, with task-specific shuffle.
    x expected shape: [N, 1, 40], y: [N]
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        task_seed: int,
        drop_last: bool = True,
        subsample_frac: float = 1.0,
        min_subsample: int = 256,
    ):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        subsample_frac = float(subsample_frac)
        if subsample_frac < 1.0:
            rng = np.random.RandomState(int(task_seed))
            n = x.shape[0]
            m = int(n * subsample_frac)
            m = max(min_subsample, m)
            m = min(n, m)
            idx = rng.choice(n, size=m, replace=False)
            x = x[idx]
            y = y[idx]

        x_t = torch.from_numpy(x)  # [N,1,40]
        y_t = torch.from_numpy(y)  # [N]

        gen = torch.Generator()
        gen.manual_seed(int(task_seed))

        self.loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=gen,
        )
        self._iter = iter(self.loader)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ---------------------- Optimizee: Conv1D (same as your F1) ----------------------
class Conv1DMNIST1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,40]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.gap(x)            # [B,64,1]
        x = x.view(x.size(0), -1)  # [B,64]
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)         # [B,10]


def init_conv1d_params(model: nn.Module, init_scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Create a dict of leaf tensors (requires_grad=True) from model's default initialized params.
    If init_scale != 1.0, scales ALL parameters multiplicatively (for experiments).
    """
    device = get_device()
    params: Dict[str, torch.Tensor] = {}
    scale = float(init_scale)

    for name, p in model.named_parameters():
        t = p.detach().to(device)
        if scale != 1.0:
            t = t * scale
        params[name] = detach_var(t)

    return params


def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Stateless forward with an explicit param dict.
    """
    if _functional_call is None:
        raise RuntimeError(
            "torch.nn.utils.stateless.functional_call is not available. "
            "Please use a newer PyTorch (>=1.13 recommended)."
        )
    return _functional_call(model, params, (x,))


# ---------------------- Learned Optimizer ----------------------
class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise 2-layer LSTM optimizer with gradient preprocessing (Appendix A).
    NO tanh bound on output (matches the common reproduction notebook).
    """

    def __init__(self, hidden_sz: int = 20, preproc: bool = True, preproc_factor: float = 10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = math.exp(-preproc_factor)

        if preproc:
            self.lstm1 = nn.LSTMCell(2, hidden_sz)
        else:
            self.lstm1 = nn.LSTMCell(1, hidden_sz)

        self.lstm2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.out = nn.Linear(hidden_sz, 1)

    def _preprocess(self, g: torch.Tensor) -> torch.Tensor:
        gd = g.data  # stop grad through gradient input
        out = torch.zeros(gd.size(0), 2, device=gd.device)
        keep = (gd.abs() >= self.preproc_threshold).squeeze()

        out[:, 0][keep] = (torch.log(gd.abs()[keep] + 1e-8) / self.preproc_factor).squeeze()
        out[:, 1][keep] = torch.sign(gd[keep]).squeeze()

        out[:, 0][~keep] = -1
        out[:, 1][~keep] = (math.exp(self.preproc_factor) * gd[~keep]).squeeze()
        return out

    def forward(self, grad: torch.Tensor, hidden: List[torch.Tensor], cell: List[torch.Tensor]):
        if self.preproc:
            grad = self._preprocess(grad)

        h0, c0 = self.lstm1(grad, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))

        update = self.out(h1)  # no tanh
        return update, (h0, h1), (c0, c1)


# ---------------------- helpers ----------------------
def _count_params(params: Dict[str, torch.Tensor]) -> int:
    return int(sum(p.numel() for p in params.values()))


def _iter_params_in_order(params: Dict[str, torch.Tensor]):
    # dict insertion order is stable in Python 3.7+, and our init preserves module order
    for k in params.keys():
        yield k, params[k]


@torch.no_grad()
def eval_params_on_dataset(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    x_np: np.ndarray,
    y_np: np.ndarray,
    batch_size: int = 512,
) -> Tuple[float, float]:
    device = get_device()
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).to(device)

    n = x.size(0)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = functional_forward(model, params, xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.size(0))

    return total_loss / max(total, 1), total_correct / max(total, 1)


def run_inner_trajectory_get_final_params(
    model: nn.Module,
    opt_net: LearnedOptimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    optim_steps: int,
    out_mul: float,
    batch_size: int,
    task_seed: int,
    init_scale: float,
    task_subsample_frac: float,
) -> Dict[str, torch.Tensor]:
    """
    Run optimizee training for optim_steps using learned optimizer (no meta-grad),
    return final params dict.
    """
    device = get_device()
    opt_net.eval()

    task = MNIST1DTask(
        x_train, y_train,
        batch_size=batch_size,
        task_seed=task_seed,
        drop_last=True,
        subsample_frac=task_subsample_frac,
    )

    params = init_conv1d_params(model, init_scale=init_scale)
    n_params = _count_params(params)
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    for _ in range(optim_steps):
        xb, yb = task.sample()
        xb = xb.to(device)
        yb = yb.to(device)

        logits = functional_forward(model, params, xb)
        loss = F.cross_entropy(logits, yb)

        grads = torch.autograd.grad(
            loss,
            list(params.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        offset = 0
        new_params: Dict[str, torch.Tensor] = {}
        new_hidden = [torch.zeros_like(h) for h in hidden]
        new_cell = [torch.zeros_like(c) for c in cell]

        for (name, p), g in zip(_iter_params_in_order(params), grads):
            sz = p.numel()
            grad_vec = g.detach().reshape(sz, 1)

            update, h_new, c_new = opt_net(
                grad_vec,
                [h[offset:offset + sz] for h in hidden],
                [c[offset:offset + sz] for c in cell],
            )

            for i in range(2):
                new_hidden[i][offset:offset + sz] = h_new[i].detach()
                new_cell[i][offset:offset + sz] = c_new[i].detach()

            new_p = (p + out_mul * update.view_as(p)).detach()
            new_params[name] = detach_var(new_p)
            offset += sz

        params = new_params
        hidden = new_hidden
        cell = new_cell

    return params


# ---------------------- Meta-training (truncated BPTT) ----------------------
def do_fit(
    model: nn.Module,
    opt_net: LearnedOptimizer,
    meta_opt: Optional[optim.Optimizer],
    x: np.ndarray,
    y: np.ndarray,
    task_seed: int,
    optim_steps: int = 100,
    unroll: int = 20,
    out_mul: float = 0.1,
    training: bool = True,
    batch_size: int = 128,
    init_scale: float = 1.0,
    task_subsample_frac: float = 1.0,
    meta_clip_norm: float = 1.0,
) -> List[float]:
    """
    One optimizee trajectory. Returns per-step optimizee losses.
    Notebook-style:
      - loss.backward(retain_graph=training) so we can backprop meta-loss every unroll steps
      - gradients are detached before feeding to opt_net (via detach_var)
    """
    device = get_device()
    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    task = MNIST1DTask(
        x, y,
        batch_size=batch_size,
        task_seed=task_seed,
        drop_last=True,
        subsample_frac=task_subsample_frac,
    )

    params = init_conv1d_params(model, init_scale=init_scale)
    n_params = _count_params(params)
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    if training and meta_opt is not None:
        meta_opt.zero_grad(set_to_none=True)

    total_loss = None
    losses_ever: List[float] = []

    for step in range(1, optim_steps + 1):
        xb, yb = task.sample()
        xb = xb.to(device)
        yb = yb.to(device)

        logits = functional_forward(model, params, xb)
        loss = F.cross_entropy(logits, yb)
        losses_ever.append(float(loss.item()))

        total_loss = loss if total_loss is None else (total_loss + loss)

        # populate grads on current params
        loss.backward(retain_graph=training)

        offset = 0
        new_params: Dict[str, torch.Tensor] = {}
        new_hidden = [torch.zeros_like(h) for h in hidden]
        new_cell = [torch.zeros_like(c) for c in cell]

        for name, p in _iter_params_in_order(params):
            sz = p.numel()
            if p.grad is None:
                raise RuntimeError(f"Gradient is None for param {name}.")
            grad_in = p.grad.reshape(sz, 1)
            grad_in = detach_var(grad_in)  # detach gradient input like notebook

            update, h_new, c_new = opt_net(
                grad_in,
                [h[offset:offset + sz] for h in hidden],
                [c[offset:offset + sz] for c in cell],
            )

            for i in range(2):
                new_hidden[i][offset:offset + sz] = h_new[i]
                new_cell[i][offset:offset + sz] = c_new[i]

            new_p = p + out_mul * update.view_as(p)
            new_p.retain_grad()
            new_params[name] = new_p
            offset += sz

        if step % unroll == 0:
            if training and meta_opt is not None:
                meta_opt.zero_grad(set_to_none=True)
                total_loss.backward()

                if meta_clip_norm is not None and float(meta_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(meta_clip_norm))

                meta_opt.step()

            params = {k: detach_var(v) for k, v in new_params.items()}
            hidden = [detach_var(h) for h in new_hidden]
            cell = [detach_var(c) for c in new_cell]
            total_loss = None
        else:
            params = new_params
            hidden = new_hidden
            cell = new_cell

    return losses_ever


# ---------------------- Outer loop + logging ----------------------
def train_l2lgdgd(args, run_dir: Path):
    device = get_device()
    set_seed(args.seed)

    model = Conv1DMNIST1D().to(device)
    opt_net = LearnedOptimizer(hidden_sz=args.hidden_sz, preproc=(not args.no_preproc)).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=args.lr)

    # Load base MNIST1D
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = _to_N40(xtr).astype(np.float32)
    xte = _to_N40(xte).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    # Apply preprocess split+norm if provided (to match F1/F2/F3)
    if args.preprocess_dir is not None:
        split, mean, std = load_split_and_norm(Path(args.preprocess_dir), args.data_seed)
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split["val_idx"], dtype=np.int64)

        x_train = apply_norm(xtr[train_idx], mean, std)
        y_train = ytr[train_idx]
        x_val = apply_norm(xtr[val_idx], mean, std)
        y_val = ytr[val_idx]
        x_test = apply_norm(xte, mean, std)
        y_test = yte
    else:
        # fallback: no split, use whole train as train, no val
        x_train, y_train = xtr, ytr
        x_val, y_val = xtr[:0], ytr[:0]
        x_test, y_test = xte, yte

    # reshape to [N,1,40]
    x_train = to_N1L(x_train, length=40)
    x_val = to_N1L(x_val, length=40) if x_val.size else x_val.reshape(0, 1, 40).astype(np.float32)
    x_test = to_N1L(x_test, length=40)

    best_eval_meta_loss = float("inf")
    best_state = None

    train_log_path = run_dir / "train_log.csv"
    curve_eval_path = run_dir / "curve_eval_mean.csv"

    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "meta_epoch",
            "elapsed_sec",
            "eval_meta_loss",
            "report_train_loss",
            "report_train_acc",
            "report_val_loss",
            "report_val_acc",
            "report_test_loss",
            "report_test_acc",
        ])

    with open(curve_eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["meta_epoch", "step", "loss_mean"])

    rng = np.random.RandomState(args.seed)

    def sample_task_seed(low: int, high: int) -> int:
        return int(rng.randint(low, high + 1))

    for ep in range(args.meta_epochs):
        # ---- meta-train ----
        for _ in range(args.inner_loops_per_epoch):
            task_seed = sample_task_seed(args.train_seed_low, args.train_seed_high)
            do_fit(
                model=model,
                opt_net=opt_net,
                meta_opt=meta_opt,
                x=x_train,
                y=y_train,
                task_seed=task_seed,
                optim_steps=args.optim_steps,
                unroll=args.unroll,
                out_mul=args.out_mul,
                training=True,
                batch_size=args.bs,
                init_scale=args.init_scale,
                task_subsample_frac=args.task_subsample_frac,
                meta_clip_norm=args.meta_clip_norm,
            )

        # ---- meta-eval loss curve on held-out task seeds ----
        eval_trajs = []
        for _ in range(args.eval_tasks):
            task_seed = sample_task_seed(args.eval_seed_low, args.eval_seed_high)
            traj = do_fit(
                model=model,
                opt_net=opt_net,
                meta_opt=None,
                x=x_train,
                y=y_train,
                task_seed=task_seed,
                optim_steps=args.optim_steps,
                unroll=args.unroll,
                out_mul=args.out_mul,
                training=False,
                batch_size=args.bs,
                init_scale=args.init_scale,
                task_subsample_frac=args.task_subsample_frac,
                meta_clip_norm=args.meta_clip_norm,
            )
            eval_trajs.append(traj)

        eval_meta_loss = float(np.mean([sum(t) for t in eval_trajs]))

        # save mean eval curve for AUC comparison
        traj_arr = np.asarray(eval_trajs, dtype=np.float32)  # [T, steps]
        mean_curve = traj_arr.mean(axis=0)
        with open(curve_eval_path, "a", newline="") as f:
            writer = csv.writer(f)
            for s, lv in enumerate(mean_curve, start=1):
                writer.writerow([ep, s, float(lv)])

        # ---- report optimizee performance ----
        rep_train_losses, rep_train_accs = [], []
        rep_val_losses, rep_val_accs = [], []
        rep_test_losses, rep_test_accs = [], []

        for _ in range(args.report_tasks):
            task_seed = sample_task_seed(args.eval_seed_low, args.eval_seed_high)
            final_params = run_inner_trajectory_get_final_params(
                model=model,
                opt_net=opt_net,
                x_train=x_train,
                y_train=y_train,
                optim_steps=args.optim_steps,
                out_mul=args.out_mul,
                batch_size=args.bs,
                task_seed=task_seed,
                init_scale=args.init_scale,
                task_subsample_frac=args.task_subsample_frac,
            )

            tr_loss, tr_acc = eval_params_on_dataset(model, final_params, x_train, y_train, batch_size=512)
            te_loss, te_acc = eval_params_on_dataset(model, final_params, x_test, y_test, batch_size=512)

            rep_train_losses.append(tr_loss)
            rep_train_accs.append(tr_acc)
            rep_test_losses.append(te_loss)
            rep_test_accs.append(te_acc)

            if x_val.shape[0] > 0:
                va_loss, va_acc = eval_params_on_dataset(model, final_params, x_val, y_val, batch_size=512)
                rep_val_losses.append(va_loss)
                rep_val_accs.append(va_acc)

        report_train_loss = float(np.mean(rep_train_losses))
        report_train_acc = float(np.mean(rep_train_accs))
        report_test_loss = float(np.mean(rep_test_losses))
        report_test_acc = float(np.mean(rep_test_accs))

        if x_val.shape[0] > 0:
            report_val_loss = float(np.mean(rep_val_losses))
            report_val_acc = float(np.mean(rep_val_accs))
        else:
            report_val_loss = float("nan")
            report_val_acc = float("nan")

        elapsed = time.time() - args._start_time

        print(
            f"[MetaEpoch {ep:03d}] "
            f"eval_meta_loss={eval_meta_loss:.6f} "
            f"train_loss={report_train_loss:.4f} train_acc={report_train_acc:.4f} "
            f"val_loss={report_val_loss:.4f} val_acc={report_val_acc:.4f} "
            f"test_loss={report_test_loss:.4f} test_acc={report_test_acc:.4f}"
        )

        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                f"{elapsed:.3f}",
                f"{eval_meta_loss:.8f}",
                f"{report_train_loss:.8f}",
                f"{report_train_acc:.6f}",
                f"{report_val_loss:.8f}",
                f"{report_val_acc:.6f}",
                f"{report_test_loss:.8f}",
                f"{report_test_acc:.6f}",
            ])

        if args.wandb and wandb is not None:
            wandb.log({
                "meta_epoch": ep,
                "eval_meta_loss": eval_meta_loss,
                "report/train_loss": report_train_loss,
                "report/train_acc": report_train_acc,
                "report/val_loss": report_val_loss,
                "report/val_acc": report_val_acc,
                "report/test_loss": report_test_loss,
                "report/test_acc": report_test_acc,
                "time/elapsed_sec": elapsed,
            })
            # log AUC proxy directly (mean_curve sum)
            wandb.log({"eval_curve/auc_sum": float(mean_curve.sum())})

        if eval_meta_loss < best_eval_meta_loss:
            best_eval_meta_loss = eval_meta_loss
            best_state = copy.deepcopy(opt_net.state_dict())
            torch.save(best_state, run_dir / "best_opt.pt")

    result = {
        "method": "mnist1d_l2lgdgd_conv1d_taskdist",
        "optimizee": "Conv1D",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "meta_epochs": int(args.meta_epochs),
        "inner_loops_per_epoch": int(args.inner_loops_per_epoch),
        "eval_tasks": int(args.eval_tasks),
        "report_tasks": int(args.report_tasks),
        "optim_steps": int(args.optim_steps),
        "unroll": int(args.unroll),
        "lr": float(args.lr),
        "bs": int(args.bs),
        "out_mul": float(args.out_mul),
        "init_scale": float(args.init_scale),
        "hidden_sz": int(args.hidden_sz),
        "preproc": bool(not args.no_preproc),
        "preprocess_dir": (str(args.preprocess_dir) if args.preprocess_dir is not None else None),
        "task_subsample_frac": float(args.task_subsample_frac),
        "meta_clip_norm": float(args.meta_clip_norm),
        "train_seed_low": int(args.train_seed_low),
        "train_seed_high": int(args.train_seed_high),
        "eval_seed_low": int(args.eval_seed_low),
        "eval_seed_high": int(args.eval_seed_high),
        "best_eval_meta_loss": float(best_eval_meta_loss),
        "device": str(device),
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_opt.pt"),
        "artifacts": {
            "train_log": str(train_log_path),
            "curve_eval_mean": str(curve_eval_path),
        },
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return best_eval_meta_loss, best_state


def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)

    ap.add_argument("--meta_epochs", type=int, default=50)
    ap.add_argument("--inner_loops_per_epoch", type=int, default=20)
    ap.add_argument("--eval_tasks", type=int, default=10)
    ap.add_argument("--report_tasks", type=int, default=3)

    ap.add_argument("--optim_steps", type=int, default=100)
    ap.add_argument("--unroll", type=int, default=20)

    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out_mul", type=float, default=0.05)

    # Conv1D init: default keep PyTorch's init (scale=1.0)
    ap.add_argument("--init_scale", type=float, default=1.0)

    ap.add_argument("--hidden_sz", type=int, default=20)
    ap.add_argument("--no_preproc", action="store_true")  # default uses preproc=True

    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/preprocess",
        help="Loads split.json + norm.json under {preprocess_dir}/seed_{data_seed}/ to align with F1/F2/F3.",
    )

    ap.add_argument("--task_subsample_frac", type=float, default=1.0)
    ap.add_argument("--meta_clip_norm", type=float, default=5.0)

    ap.add_argument("--train_seed_low", type=int, default=0)
    ap.add_argument("--train_seed_high", type=int, default=9999)
    ap.add_argument("--eval_seed_low", type=int, default=10000)
    ap.add_argument("--eval_seed_high", type=int, default=10999)

    ap.add_argument("--runs_dir", type=str, default="runs")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-online")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2lgdgd_conv1d")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    return ap


def main():
    args = build_argparser().parse_args()

    print(f"[INFO] device = {get_device()}")

    run_name = (
        f"mnist1d_l2lgdgd_conv1d_seed{args.seed}_data{args.data_seed}_"
        f"sub{args.task_subsample_frac}_clip{args.meta_clip_norm}_"
        f"steps{args.optim_steps}_unroll{args.unroll}_"
        f"lr{args.lr}_out{args.out_mul}_initscale{args.init_scale}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    args._start_time = time.time()

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    best_loss, _ = train_l2lgdgd(args, run_dir)
    elapsed = time.time() - args._start_time
    print(f"[DONE] best eval_meta_loss={best_loss:.6f} elapsed={elapsed/60:.2f} min")

    if wandb_run is not None:
        wandb_run.summary["best_eval_meta_loss"] = float(best_loss)
        wandb_run.summary["elapsed_sec"] = float(elapsed)
        for p in [
            run_dir / "train_log.csv",
            run_dir / "curve_eval_mean.csv",
            run_dir / "result.json",
            run_dir / "best_opt.pt",
        ]:
            if p.exists():
                wandb.save(str(p))
        wandb_run.finish()


if __name__ == "__main__":
    main()

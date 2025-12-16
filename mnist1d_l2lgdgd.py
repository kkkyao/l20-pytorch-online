#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learning to Learn by Gradient Descent by Gradient Descent (L2L-GDGD)
Paper-style coordinate-wise LSTM learned optimizer, with MLP optimizee (recommended for MNIST1D).

Alignment to your baseline/F1/F2/F3 pipeline:
  - Uses {preprocess_dir}/seed_{data_seed}/split.json and norm.json
  - Train uses split["train_idx"], Val uses split["val_idx"], Test uses xte
  - Per-position normalization (x - mean) / std
  - Reports train/val/test loss+acc per meta-epoch
  - Logs mean eval loss curve per meta-epoch for AUC comparison

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


# ---------------------- Task: minibatch stream (task distribution) ----------------------
class MNIST1DTask:
    """
    A task is a minibatch stream over a (subsampled) dataset, with task-specific shuffle.
    x expected shape: [N, 40], y: [N]
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

        x_t = torch.from_numpy(x)  # [N,40]
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


# ---------------------- Optimizee: MLP (paper-style) ----------------------
class MLPOptimizee(nn.Module):
    """
    MLP optimizee:
      input(40) -> hidden(hidden_dim, sigmoid) -> output(10)

    We store parameters in a dict of leaf tensors, to support meta-learning updates.
    """

    def __init__(
        self,
        hidden_dim: int = 20,
        init_scale: float = 1e-3,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        device = get_device()
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)

        if params is None:
            p = {
                "W1": torch.randn(40, self.hidden_dim, device=device) * self.init_scale,
                "b1": torch.zeros(self.hidden_dim, device=device),
                "W2": torch.randn(self.hidden_dim, 10, device=device) * self.init_scale,
                "b2": torch.zeros(10, device=device),
            }
            self.params = {k: detach_var(v) for k, v in p.items()}
        else:
            self.params = params

    def all_named_parameters(self):
        return list(self.params.items())

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(x @ self.params["W1"] + self.params["b1"])
        return h @ self.params["W2"] + self.params["b2"]

    def forward_with_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return F.cross_entropy(logits, y)

    def forward(self, task: MNIST1DTask) -> torch.Tensor:
        x, y = task.sample()
        x = x.to(get_device())
        y = y.to(get_device())
        return self.forward_with_batch(x, y)


# ---------------------- Learned Optimizer ----------------------
class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise 2-layer LSTM optimizer with gradient preprocessing (Appendix A).
    NO tanh bound on output (matches common reproduction notebooks).
    """

    def __init__(self, hidden_sz: int = 20, preproc: bool = True, preproc_factor: float = 10.0):
        super().__init__()
        self.hidden_sz = int(hidden_sz)
        self.preproc = bool(preproc)
        self.preproc_factor = float(preproc_factor)
        self.preproc_threshold = math.exp(-self.preproc_factor)

        if self.preproc:
            self.lstm1 = nn.LSTMCell(2, self.hidden_sz)
        else:
            self.lstm1 = nn.LSTMCell(1, self.hidden_sz)

        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz, 1)

    def _preprocess(self, g: torch.Tensor) -> torch.Tensor:
        # stop grad through gradient input (paper / common ref behavior)
        gd = g.data
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
def _count_params_from_optimizee(optimizee: MLPOptimizee) -> int:
    return int(sum(p.numel() for _, p in optimizee.all_named_parameters()))


@torch.no_grad()
def eval_params_on_dataset(
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

        h = torch.sigmoid(xb @ params["W1"] + params["b1"])
        logits = h @ params["W2"] + params["b2"]

        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.size(0))

    return total_loss / max(total, 1), total_correct / max(total, 1)


def run_inner_trajectory_get_final_params(
    opt_net: LearnedOptimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    optim_steps: int,
    out_mul: float,
    batch_size: int,
    task_seed: int,
    init_scale: float,
    hidden_dim: int,
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

    optimizee = MLPOptimizee(hidden_dim=hidden_dim, init_scale=init_scale)
    params = {k: v for k, v in optimizee.params.items()}

    n_params = int(sum(p.numel() for p in params.values()))
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    for _ in range(optim_steps):
        xb, yb = task.sample()
        xb = xb.to(device)
        yb = yb.to(device)

        h = torch.sigmoid(xb @ params["W1"] + params["b1"])
        logits = h @ params["W2"] + params["b2"]
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
        new_hidden = [torch.zeros_like(hh) for hh in hidden]
        new_cell = [torch.zeros_like(cc) for cc in cell]

        for (name, p), g in zip(params.items(), grads):
            sz = p.numel()
            grad_vec = g.detach().reshape(sz, 1)

            update, h_new, c_new = opt_net(
                grad_vec,
                [hh[offset:offset + sz] for hh in hidden],
                [cc[offset:offset + sz] for cc in cell],
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
    init_scale: float = 1e-3,
    hidden_dim: int = 20,
    task_subsample_frac: float = 1.0,
    meta_clip_norm: float = 1.0,
) -> List[float]:
    """
    One optimizee trajectory. Returns per-step optimizee losses.

    FIXED semantics:
      - Per-step gradients are computed via torch.autograd.grad (NO loss.backward for grads).
      - Meta-loss backward happens ONLY at unroll boundaries (NO "backward then backward").
    """
    device = get_device()
    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1  # evaluation: no truncated BPTT segments

    task = MNIST1DTask(
        x, y,
        batch_size=batch_size,
        task_seed=task_seed,
        drop_last=True,
        subsample_frac=task_subsample_frac,
    )

    optimizee = MLPOptimizee(hidden_dim=hidden_dim, init_scale=init_scale)
    n_params = _count_params_from_optimizee(optimizee)

    # hidden/cell are coordinate-wise states, shape [num_params, hidden_sz]
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    if training and meta_opt is not None:
        meta_opt.zero_grad(set_to_none=True)

    total_loss: Optional[torch.Tensor] = None
    losses_ever: List[float] = []

    for step in range(1, optim_steps + 1):
        # ---- forward loss ----
        loss = optimizee(task)
        losses_ever.append(float(loss.item()))
        total_loss = loss if total_loss is None else (total_loss + loss)

        # ---- per-step grads: autograd.grad (NOT backward) ----
        param_items = optimizee.all_named_parameters()
        param_list = [p for _, p in param_items]

        # retain_graph is needed in training because we will backward(total_loss) later at unroll boundary
        grads = torch.autograd.grad(
            loss,
            param_list,
            create_graph=False,              # paper/common ref: do not backprop through grads
            retain_graph=bool(training),     # keep graph until meta backward
            allow_unused=False,
        )

        # ---- apply learned optimizer update ----
        offset = 0
        new_params: Dict[str, torch.Tensor] = {}
        new_hidden = [torch.zeros_like(hh) for hh in hidden]
        new_cell = [torch.zeros_like(cc) for cc in cell]

        for (name, p), g in zip(param_items, grads):
            sz = p.numel()

            # gradient input is detached (paper/common ref); meta-grad flows through opt_net params via update
            grad_in = g.detach().reshape(sz, 1)

            update, h_new, c_new = opt_net(
                grad_in,
                [hh[offset:offset + sz] for hh in hidden],
                [cc[offset:offset + sz] for cc in cell],
            )

            for i in range(2):
                new_hidden[i][offset:offset + sz] = h_new[i]
                new_cell[i][offset:offset + sz] = c_new[i]

            new_p = p + out_mul * update.view_as(p)
            new_params[name] = new_p
            offset += sz

        # ---- unroll boundary: do ONE meta backward and step ----
        if step % unroll == 0:
            if training and meta_opt is not None:
                meta_opt.zero_grad(set_to_none=True)
                assert total_loss is not None
                total_loss.backward()

                if meta_clip_norm is not None and float(meta_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(meta_clip_norm))

                meta_opt.step()

            # truncate BPTT: detach params + states into fresh leaves for next segment
            if training:
                optimizee = MLPOptimizee(
                    hidden_dim=hidden_dim,
                    init_scale=init_scale,
                    params={k: detach_var(v) for k, v in new_params.items()},
                )
                hidden = [hh.detach() for hh in new_hidden]
                cell = [cc.detach() for cc in new_cell]
            else:
                # eval: always detach each step/segment
                optimizee = MLPOptimizee(
                    hidden_dim=hidden_dim,
                    init_scale=init_scale,
                    params={k: detach_var(v.detach()) for k, v in new_params.items()},
                )
                hidden = [hh.detach() for hh in new_hidden]
                cell = [cc.detach() for cc in new_cell]

            total_loss = None
        else:
            # within unroll: keep graph
            if training:
                optimizee = MLPOptimizee(hidden_dim=hidden_dim, init_scale=init_scale, params=new_params)
                hidden = new_hidden
                cell = new_cell
            else:
                # eval: we set unroll=1 above, so should not reach here
                optimizee = MLPOptimizee(
                    hidden_dim=hidden_dim,
                    init_scale=init_scale,
                    params={k: detach_var(v.detach()) for k, v in new_params.items()},
                )
                hidden = [hh.detach() for hh in new_hidden]
                cell = [cc.detach() for cc in new_cell]

    return losses_ever


# ---------------------- Outer loop + logging ----------------------
def train_l2lgdgd(args, run_dir: Path):
    device = get_device()
    set_seed(args.seed)

    opt_net = LearnedOptimizer(hidden_sz=args.hidden_sz, preproc=(not args.no_preproc)).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=args.lr)

    # Load base MNIST1D
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = _to_N40(xtr).astype(np.float32)
    xte = _to_N40(xte).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    # Apply preprocess split+norm (to align with baselines/F*)
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
        x_train, y_train = xtr, ytr
        x_val, y_val = xtr[:0], ytr[:0]
        x_test, y_test = xte, yte

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
                hidden_dim=args.hidden_dim,
                task_subsample_frac=args.task_subsample_frac,
                meta_clip_norm=args.meta_clip_norm,
            )

        # ---- meta-eval loss curve on held-out task seeds ----
        eval_trajs = []
        for _ in range(args.eval_tasks):
            task_seed = sample_task_seed(args.eval_seed_low, args.eval_seed_high)
            traj = do_fit(
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
                hidden_dim=args.hidden_dim,
                task_subsample_frac=args.task_subsample_frac,
                meta_clip_norm=args.meta_clip_norm,
            )
            eval_trajs.append(traj)

        eval_meta_loss = float(np.mean([sum(t) for t in eval_trajs]))

        traj_arr = np.asarray(eval_trajs, dtype=np.float32)  # [eval_tasks, steps]
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
                opt_net=opt_net,
                x_train=x_train,
                y_train=y_train,
                optim_steps=args.optim_steps,
                out_mul=args.out_mul,
                batch_size=args.bs,
                task_seed=task_seed,
                init_scale=args.init_scale,
                hidden_dim=args.hidden_dim,
                task_subsample_frac=args.task_subsample_frac,
            )

            tr_loss, tr_acc = eval_params_on_dataset(final_params, x_train, y_train, batch_size=512)
            te_loss, te_acc = eval_params_on_dataset(final_params, x_test, y_test, batch_size=512)
            rep_train_losses.append(tr_loss)
            rep_train_accs.append(tr_acc)
            rep_test_losses.append(te_loss)
            rep_test_accs.append(te_acc)

            if x_val.shape[0] > 0:
                va_loss, va_acc = eval_params_on_dataset(final_params, x_val, y_val, batch_size=512)
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
                "eval_curve/auc_sum": float(mean_curve.sum()),
            })

        if eval_meta_loss < best_eval_meta_loss:
            best_eval_meta_loss = eval_meta_loss
            best_state = copy.deepcopy(opt_net.state_dict())
            torch.save(best_state, run_dir / "best_opt.pt")

    result = {
        "method": "mnist1d_l2lgdgd_mlp",
        "optimizee": f"MLP(40->{args.hidden_dim}(sigmoid)->10)",
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
        "hidden_dim": int(args.hidden_dim),
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
    ap.add_argument("--out_mul", type=float, default=0.1)

    ap.add_argument("--init_scale", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=20)

    ap.add_argument("--hidden_sz", type=int, default=20)
    ap.add_argument("--no_preproc", action="store_true")

    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/preprocess",
        help="Loads split.json + norm.json under {preprocess_dir}/seed_{data_seed}/.",
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
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2lgdgd_mlp")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    return ap


def main():
    args = build_argparser().parse_args()

    print(f"[INFO] device = {get_device()}")

    run_name = (
        f"mnist1d_l2lgdgd_mlp_seed{args.seed}_data{args.data_seed}_"
        f"sub{args.task_subsample_frac}_clip{args.meta_clip_norm}_"
        f"steps{args.optim_steps}_unroll{args.unroll}_"
        f"lr{args.lr}_out{args.out_mul}_initscale{args.init_scale}_"
        f"hd{args.hidden_dim}_"
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

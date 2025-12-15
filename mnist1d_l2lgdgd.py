#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learning to Learn by Gradient Descent by Gradient Descent (L2L-GDGD)
(Task-distribution version, paper/notebook-style)

This version aligns closer to the reproduction notebook you pasted:

Key changes vs your current file:
  - REMOVE tanh bounding on optimizer output (notebook doesn't have tanh)
  - Optimizee init scale default 1e-3 (notebook uses 1e-3 for MNIST matrices)
  - Add MNIST1D per-position normalization via --preprocess_dir/norm.json (high impact)
  - Keep "task = minibatch stream" distribution: fixed dataset by data_seed, task differs by shuffle order (task_seed)

Adds/keeps:
  - argparse
  - wandb optional
  - run_dir outputs
  - per-meta-epoch evaluation of optimizee train/test loss+acc (like F1/F2/F3)
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
    This is CRITICAL for truncated BPTT in L2L-GDGD.
    """
    device = get_device()
    out = v.detach().clone().to(device)
    out.requires_grad_(True)
    out.retain_grad()
    return out


# ---------------------- MNIST1D normalization helpers ----------------------
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


def load_norm_stats(preprocess_dir: Path, data_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expect: {preprocess_dir}/seed_{data_seed}/norm.json
      {"mean": [...len40...], "std": [...len40...]}
    """
    pdir = preprocess_dir / f"seed_{data_seed}"
    norm_path = pdir / "norm.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"norm.json not found: {norm_path}")
    with open(norm_path, "r") as f:
        norm = json.load(f)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (x - mean[None, :]) / (std[None, :] + eps)


# ---------------------- Task: minibatch stream (paper-style) ----------------------
class MNIST1DTask:
    """
    A "task" is a minibatch stream over a fixed dataset, with its own shuffle order
    controlled by task_seed.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        task_seed: int,
        drop_last: bool = True,
    ):
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
        self._iter = iter(self.loader)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ---------------------- Optimizee (paper-aligned MLP) ----------------------
class MNIST1DOptimizee(nn.Module):
    """
    Functional MLP optimizee:
        input(40) -> hidden(20, sigmoid) -> output(10)

    NOTE:
      - Parameters stored in a plain dict are NOT auto-moved by .to(device).
      - So when params is None, we MUST create them on the correct device.
      - init_scale default is 1e-3 to match the notebook MNIST net style (important for sigmoid).
    """

    def __init__(
        self,
        hidden_dim: int = 20,
        init_scale: float = 1e-3,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        device = get_device()
        self.hidden_dim = hidden_dim
        self.init_scale = float(init_scale)

        if params is None:
            self.params = {
                "W1": torch.randn(40, hidden_dim, device=device) * self.init_scale,
                "b1": torch.zeros(hidden_dim, device=device),
                "W2": torch.randn(hidden_dim, 10, device=device) * self.init_scale,
                "b2": torch.zeros(10, device=device),
            }
            # make them leaf tensors with grad
            for k in list(self.params.keys()):
                self.params[k] = detach_var(self.params[k])
        else:
            self.params = params  # assume already on correct device and requires_grad

    def all_named_parameters(self):
        return list(self.params.items())

    def forward_with_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(x @ self.params["W1"] + self.params["b1"])
        logits = h @ self.params["W2"] + self.params["b2"]
        return F.cross_entropy(logits, y)

    def forward(self, task: MNIST1DTask) -> torch.Tensor:
        x, y = task.sample()
        x = x.to(get_device())
        y = y.to(get_device())
        return self.forward_with_batch(x, y)


# ---------------------- Learned Optimizer ----------------------
class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise LSTM optimizer (paper-aligned), with gradient preprocessing.

    IMPORTANT: notebook version DOES NOT tanh-bound the output.
    So we return raw linear output as update.
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
        # Detach path like notebook: use .data to avoid gradient flow through gradient input.
        gd = g.data
        out = torch.zeros(gd.size(0), 2, device=gd.device)
        keep = (gd.abs() >= self.preproc_threshold).squeeze()

        out[:, 0][keep] = (torch.log(gd.abs()[keep] + 1e-8) / self.preproc_factor).squeeze()
        out[:, 1][keep] = torch.sign(gd[keep]).squeeze()

        out[:, 0][~keep] = -1
        out[:, 1][~keep] = (math.exp(self.preproc_factor) * gd[~keep]).squeeze()

        return out  # plain tensor is fine

    def forward(
        self,
        grad: torch.Tensor,
        hidden: List[torch.Tensor],
        cell: List[torch.Tensor],
    ):
        if self.preproc:
            grad = self._preprocess(grad)

        h0, c0 = self.lstm1(grad, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))

        update = self.out(h1)  # NO tanh
        return update, (h0, h1), (c0, c1)


# ---------------------- helpers ----------------------
def _count_params(optimizee: MNIST1DOptimizee) -> int:
    return sum(int(np.prod(p.size())) for _, p in optimizee.all_named_parameters())


def forward_logits_with_params(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    h = torch.sigmoid(x @ params["W1"] + params["b1"])
    logits = h @ params["W2"] + params["b2"]
    return logits


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
        logits = forward_logits_with_params(xb, params)
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
) -> Dict[str, torch.Tensor]:
    """
    Run optimizee training for optim_steps using learned optimizer (no meta-grad),
    return final params dict. Used for reporting train/test metrics per meta-epoch.
    """
    device = get_device()
    opt_net.eval()

    task = MNIST1DTask(x_train, y_train, batch_size=batch_size, task_seed=task_seed, drop_last=True)
    optimizee = MNIST1DOptimizee(init_scale=init_scale)
    params = {k: v for k, v in optimizee.params.items()}  # on device already

    n_params = sum(int(np.prod(p.size())) for p in params.values())
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    for _ in range(optim_steps):
        x, y = task.sample()
        x = x.to(device)
        y = y.to(device)

        logits = forward_logits_with_params(x, params)
        loss = F.cross_entropy(logits, y)

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

        for (name, p), g in zip(params.items(), grads):
            sz = int(np.prod(p.size()))
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
            new_p = detach_var(new_p)
            new_params[name] = new_p

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
) -> List[float]:
    """
    One optimizee trajectory (one task seed). Returns per-step optimizee losses.
    Notebook-style: uses loss.backward(...) to populate p.grad and then uses p.grad as input.
    """
    device = get_device()
    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    task = MNIST1DTask(x, y, batch_size=batch_size, task_seed=task_seed, drop_last=True)
    optimizee = MNIST1DOptimizee(init_scale=init_scale)

    n_params = _count_params(optimizee)
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    if training and meta_opt is not None:
        meta_opt.zero_grad(set_to_none=True)

    total_loss = None
    losses_ever: List[float] = []

    for step in range(1, optim_steps + 1):
        loss = optimizee(task)
        losses_ever.append(float(loss.item()))

        total_loss = loss if total_loss is None else (total_loss + loss)

        # populate grads on current optimizee params
        loss.backward(retain_graph=training)

        offset = 0
        new_params: Dict[str, torch.Tensor] = {}
        new_hidden = [torch.zeros_like(h) for h in hidden]
        new_cell = [torch.zeros_like(c) for c in cell]

        for name, p in optimizee.all_named_parameters():
            sz = int(np.prod(p.size()))
            if p.grad is None:
                raise RuntimeError(f"Gradient is None for param {name}.")
            grad_in = p.grad.view(sz, 1)
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
                meta_opt.step()

            optimizee = MNIST1DOptimizee(
                init_scale=init_scale,
                params={k: detach_var(v) for k, v in new_params.items()},
            )
            hidden = [detach_var(h) for h in new_hidden]
            cell = [detach_var(c) for c in new_cell]
            total_loss = None
        else:
            optimizee = MNIST1DOptimizee(init_scale=init_scale, params=new_params)
            hidden = new_hidden
            cell = new_cell

    return losses_ever


# ---------------------- Outer loop + logging ----------------------
def train_l2lgdgd(args, run_dir: Path):
    device = get_device()
    set_seed(args.seed)

    opt_net = LearnedOptimizer(hidden_sz=args.hidden_sz, preproc=(not args.no_preproc)).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=args.lr)

    # FIXED dataset: tasks differ only by batch stream (shuffle order)
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = _to_N40(xtr).astype(np.float32)
    xte = _to_N40(xte).astype(np.float32)

    # Optional normalization (recommended for MNIST1D)
    if args.preprocess_dir is not None:
        mean, std = load_norm_stats(Path(args.preprocess_dir), args.data_seed)
        xtr = apply_norm(xtr, mean, std)
        xte = apply_norm(xte, mean, std)

    best_eval_meta_loss = float("inf")
    best_state = None

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "meta_epoch",
            "elapsed_sec",
            "eval_meta_loss",
            "report_train_loss",
            "report_train_acc",
            "report_test_loss",
            "report_test_acc",
        ])

    # helper: sample task seeds (controls shuffle order)
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
                x=xtr,
                y=ytr,
                task_seed=task_seed,
                optim_steps=args.optim_steps,
                unroll=args.unroll,
                out_mul=args.out_mul,
                training=True,
                batch_size=args.bs,
                init_scale=args.init_scale,
            )

        # ---- meta-eval loss on held-out task seeds ----
        eval_losses = []
        for _ in range(args.eval_tasks):
            task_seed = sample_task_seed(args.eval_seed_low, args.eval_seed_high)
            traj = do_fit(
                opt_net=opt_net,
                meta_opt=None,
                x=xtr,
                y=ytr,
                task_seed=task_seed,
                optim_steps=args.optim_steps,
                unroll=args.unroll,
                out_mul=args.out_mul,
                training=False,
                batch_size=args.bs,
                init_scale=args.init_scale,
            )
            eval_losses.append(sum(traj))
        eval_meta_loss = float(np.mean(eval_losses))

        # ---- report optimizee performance (average over report_tasks) ----
        rep_train_losses, rep_train_accs = [], []
        rep_test_losses, rep_test_accs = [], []

        for _ in range(args.report_tasks):
            task_seed = sample_task_seed(args.eval_seed_low, args.eval_seed_high)
            final_params = run_inner_trajectory_get_final_params(
                opt_net=opt_net,
                x_train=xtr,
                y_train=ytr,
                optim_steps=args.optim_steps,
                out_mul=args.out_mul,
                batch_size=args.bs,
                task_seed=task_seed,
                init_scale=args.init_scale,
            )
            tr_loss, tr_acc = eval_params_on_dataset(final_params, xtr, ytr, batch_size=512)
            te_loss, te_acc = eval_params_on_dataset(final_params, xte, yte, batch_size=512)
            rep_train_losses.append(tr_loss); rep_train_accs.append(tr_acc)
            rep_test_losses.append(te_loss); rep_test_accs.append(te_acc)

        report_train_loss = float(np.mean(rep_train_losses))
        report_train_acc = float(np.mean(rep_train_accs))
        report_test_loss = float(np.mean(rep_test_losses))
        report_test_acc = float(np.mean(rep_test_accs))

        elapsed = time.time() - args._start_time

        print(
            f"[MetaEpoch {ep:03d}] "
            f"eval_meta_loss={eval_meta_loss:.6f} "
            f"report_train_loss={report_train_loss:.4f} report_train_acc={report_train_acc:.4f} "
            f"report_test_loss={report_test_loss:.4f} report_test_acc={report_test_acc:.4f}"
        )

        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                f"{elapsed:.3f}",
                f"{eval_meta_loss:.8f}",
                f"{report_train_loss:.8f}",
                f"{report_train_acc:.6f}",
                f"{report_test_loss:.8f}",
                f"{report_test_acc:.6f}",
            ])

        if args.wandb and wandb is not None:
            wandb.log({
                "meta_epoch": ep,
                "eval_meta_loss": eval_meta_loss,
                "report/train_loss": report_train_loss,
                "report/train_acc": report_train_acc,
                "report/test_loss": report_test_loss,
                "report/test_acc": report_test_acc,
                "time/elapsed_sec": elapsed,
            })

        if eval_meta_loss < best_eval_meta_loss:
            best_eval_meta_loss = eval_meta_loss
            best_state = copy.deepcopy(opt_net.state_dict())
            torch.save(best_state, run_dir / "best_opt.pt")

    result = {
        "method": "mnist1d_l2lgdgd_taskdist_stream_notebooklike",
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
        "train_seed_low": int(args.train_seed_low),
        "train_seed_high": int(args.train_seed_high),
        "eval_seed_low": int(args.eval_seed_low),
        "eval_seed_high": int(args.eval_seed_high),
        "best_eval_meta_loss": float(best_eval_meta_loss),
        "device": str(device),
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_opt.pt"),
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

    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out_mul", type=float, default=0.1)

    # notebook-like knobs
    ap.add_argument("--init_scale", type=float, default=1e-3)     # 0.001
    ap.add_argument("--hidden_sz", type=int, default=20)
    ap.add_argument("--no_preproc", action="store_true")          # default uses preproc=True

    # IMPORTANT: MNIST1D normalization (recommended)
    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default=None,
        help="If set, loads {preprocess_dir}/seed_{data_seed}/norm.json and normalizes x.",
    )

    # task seed ranges (affect only minibatch order)
    ap.add_argument("--train_seed_low", type=int, default=0)
    ap.add_argument("--train_seed_high", type=int, default=9999)
    ap.add_argument("--eval_seed_low", type=int, default=10000)
    ap.add_argument("--eval_seed_high", type=int, default=10999)

    ap.add_argument("--runs_dir", type=str, default="runs")

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-online")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2lgdgd")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    return ap


def main():
    args = build_argparser().parse_args()

    print(f"[INFO] device = {get_device()}")

    run_name = (
        f"mnist1d_l2lgdgd_seed{args.seed}_data{args.data_seed}_"
        f"streamTask_train{args.train_seed_low}-{args.train_seed_high}_"
        f"eval{args.eval_seed_low}-{args.eval_seed_high}_"
        f"lr{args.lr}_init{args.init_scale}_"
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
            config={
                "seed": args.seed,
                "data_seed": args.data_seed,
                "meta_epochs": args.meta_epochs,
                "inner_loops_per_epoch": args.inner_loops_per_epoch,
                "eval_tasks": args.eval_tasks,
                "report_tasks": args.report_tasks,
                "optim_steps": args.optim_steps,
                "unroll": args.unroll,
                "lr": args.lr,
                "bs": args.bs,
                "out_mul": args.out_mul,
                "init_scale": args.init_scale,
                "hidden_sz": args.hidden_sz,
                "preproc": (not args.no_preproc),
                "preprocess_dir": args.preprocess_dir,
                "train_seed_low": args.train_seed_low,
                "train_seed_high": args.train_seed_high,
                "eval_seed_low": args.eval_seed_low,
                "eval_seed_high": args.eval_seed_high,
            },
        )

    best_loss, _ = train_l2lgdgd(args, run_dir)
    elapsed = time.time() - args._start_time
    print(f"[DONE] best eval_meta_loss={best_loss:.6f} elapsed={elapsed/60:.2f} min")

    if wandb_run is not None:
        wandb_run.summary["best_eval_meta_loss"] = float(best_loss)
        wandb_run.summary["elapsed_sec"] = float(elapsed)
        for p in [run_dir / "train_log.csv", run_dir / "result.json", run_dir / "best_opt.pt"]:
            if p.exists():
                wandb.save(str(p))
        wandb_run.finish()


if __name__ == "__main__":
    main()

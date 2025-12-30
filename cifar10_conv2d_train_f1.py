#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BF-style Step-3 (B+F1) for CIFAR-10 + Conv2D -- ONLINE TRAIN (PyTorch)
-----------------------------------------------------------------------

This script trains a Conv2D classifier on CIFAR-10 while *simultaneously*
learning a scalar step-size rule L_theta via online meta-learning (BF1-PT style).

F1 feature:
  phi_t = log ||g_t||

Step-size:
  eta_t = c_base / (L_theta(phi_t) + eps)

Online meta-learning loop (single task):
  For each train batch:
    1) Compute train loss and gradient g_t on train batch
    2) Sample K val batches, compute val_loss and grad_val
    3) Compute dot = <grad_val, g_t> (averaged over K to reduce noise)
    4) Define meta_loss = val_loss - eta * dot + small regularizer
    5) Take one gradient step on theta (parameters of LearnedL)
    6) With the updated LearnedL, recompute eta and update w
       using a plain SGD-like step (no momentum) with global-norm clipping

Stability additions (minimal invasive):
  - Average meta dot over multiple val batches (--val_meta_batches)
  - Limit per-step eta relative change (--eta_change_ratio)
  - Optional weight decay in the manual SGD update (--wd)
  - Per-epoch eta/L/phi/dot statistics logged to eta_stats_f1.csv

There is NO separate meta-train / meta-test split in this script:
  - Conv2D backbone (w) and LearnedL (theta) are trained jointly online
    on the same CIFAR-10 task.

Data:
  - Uses CIFAR-10 loaded via data_cifar10.load_cifar10() as numpy arrays
  - Uses preprocess artifacts under:
      preprocess_dir/seed_{data_seed}/{split.json, norm.json}
    to define train/val split and per-channel normalization.

Outputs:
  run_dir = runs/<run_name> with:
    - curve_f1.csv           (per-batch train loss)
    - mechanism_f1.csv       (eta_t, L_theta, phi_t per step)
    - eta_stats_f1.csv       (per-epoch eta stats)
    - train_log_f1.csv       (per-epoch train/val/test loss+acc)
    - time_log_f1.csv        (per-epoch timing)
    - result_f1.json         (final summary)
  Optional: logs to Weights & Biases if --wandb is enabled.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_cifar10 import load_cifar10  # your numpy CIFAR-10 loader
from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader


# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# =============================== Utilities ===============================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_artifacts(preprocess_dir: Path, seed: int):
    """
    Load split.json and norm.json from:
      preprocess_dir/seed_{seed}/
    """
    pdir = preprocess_dir / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    meta_path = pdir / "meta.json"

    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifacts in: {pdir}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return split, norm, meta


def to_nchw_and_norm(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert CIFAR-10 images to [N, C, H, W] and apply per-channel normalization.

    Input:
      x: [N, 32, 32, 3] or [N, 3, 32, 32] in float32, typically in [0, 1]
      mean, std: per-channel vectors of length 3 (computed on [0,1] scale)

    Returns:
      x_norm: float32 array [N, 3, 32, 32]
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 4:
        raise AssertionError(
            f"Expected CIFAR-10 images with 4 dimensions, got shape {x.shape}"
        )

    # Ensure NCHW layout
    if x.shape[1] == 3:
        x_nchw = x
    elif x.shape[-1] == 3:
        x_nchw = np.transpose(x, (0, 3, 1, 2))
    else:
        raise AssertionError(
            f"Unexpected CIFAR-10 shape {x.shape}, cannot infer channel dimension."
        )

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    if mean.ndim == 1:
        if mean.shape[0] != x_nchw.shape[1]:
            raise AssertionError(
                f"mean length {mean.shape[0]} does not match channels {x_nchw.shape[1]}"
            )
        mean = mean.reshape(1, -1, 1, 1)
    if std.ndim == 1:
        if std.shape[0] != x_nchw.shape[1]:
            raise AssertionError(
                f"std length {std.shape[0]} does not match channels {x_nchw.shape[1]}"
            )
        std = std.reshape(1, -1, 1, 1)

    x_norm = (x_nchw - mean) / std
    return x_norm


# =========================== Conv2D Backbone ============================

class Conv2DBaseline(nn.Module):
    """
    Simple VGG-like ConvNet used as a CIFAR-10 baseline.
    This architecture matches the Conv2D backbone used in F1/F2/F3 experiments.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================ Learned L_theta ============================

class LearnedL(nn.Module):
    """
    Learned scalar L_theta for F1 feature:
      phi = log ||g||
    Output L_theta in [L_min, L_max].

    Input:
      phi: [B, 1]
    Output:
      L_theta: [B, 1]
    """

    def __init__(self, L_min: float = 1e-3, L_max: float = 1e3, hidden: int = 32):
        super().__init__()
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.h1 = nn.Linear(1, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        z = self.act(self.h1(phi))
        z = self.act(self.h2(z))
        s = self.sigmoid(self.out(z))  # [B,1] in (0,1)
        L = self.L_min + (self.L_max - self.L_min) * s
        return L


# ========================== Batch Loss Logger ===========================

class BatchLossLogger:
    """
    Log per-batch training loss.

    Writes CSV: curve_f1.csv with schema:
      iter,epoch,loss,method,seed,opt,lr
    """

    def __init__(self, run_dir: Path, meta: dict, flush_every: int = 200):
        self.run_dir = Path(run_dir)
        self.meta = meta
        self.flush_every = flush_every
        self.global_iter = 0
        self.curr_epoch = 0
        self.rows = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.curve_path = self.run_dir / "curve_f1.csv"
        with open(self.curve_path, "w") as f:
            f.write("iter,epoch,loss,method,seed,opt,lr\n")

    def on_epoch_begin(self, epoch: int):
        self.curr_epoch = int(epoch)

    def on_train_batch_end(self, loss_value: float):
        row = [
            self.global_iter,
            self.curr_epoch,
            float(loss_value),
            str(self.meta.get("method", "unknown")).lower(),
            int(self.meta.get("seed", -1)),
            str(self.meta.get("opt", "unknown")).lower(),
            float(self.meta.get("lr", float("nan"))),
        ]
        self.rows.append(row)
        self.global_iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def on_train_end(self):
        if self.rows:
            self._flush()

    def _flush(self):
        with open(self.curve_path, "a") as f:
            for r in self.rows:
                f.write(",".join(map(str, r)) + "\n")
        self.rows.clear()


# ========================= Helper: Evaluation ===========================

def evaluate_on_loader(model, device, loader, criterion):
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


# ================================ Main =================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--data_seed", type=int, default=42, help="data split / preprocess seed")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/cifar10_conv2d_preprocess",
        help="directory containing CIFAR-10 preprocess artifacts",
    )

    # Step-size learner hyperparameters (BF1-style)
    ap.add_argument("--c_base", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--Lmin", type=float, default=1e-3)
    ap.add_argument("--Lmax", type=float, default=1e3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--eta_max", type=float, default=1.0)
    ap.add_argument("--theta_lr", type=float, default=1e-3)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # --- Stability / regularization knobs ---
    ap.add_argument("--val_meta_batches", type=int, default=2,
                    help="number of val mini-batches to average for meta dot (reduces noise)")
    ap.add_argument("--eta_change_ratio", type=float, default=0.05,
                    help="max relative change of eta per step, e.g. 0.05 means +/-5%")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay applied in the manual w update (0 disables)")

    # WandB logging
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="l2o-cifar10", help="WandB project name")
    ap.add_argument("--wandb_group", type=str, default="cifar10_conv2d_f1_bf1pt", help="WandB run name, defaults to run_name")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="optional WandB run name, defaults to run_name")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    run_name = (
        f"cifar10_conv2d_f1_data{args.data_seed}_seed{args.seed}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    # ------------------------------------------------------------------
    # WandB init (optional)
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError(
                "wandb is not installed, but --wandb was passed. "
                "Install via `pip install wandb` or disable --wandb."
            )
        wandb_config = {
            "stage": "online-train",
            "dataset": "CIFAR-10",
            "backbone": "Conv2D",
            "method": "learned_l_f1_online_bf1pt",
            "seed": args.seed,
            "data_seed": args.data_seed,
            "epochs": args.epochs,
            "batch_size": args.bs,
            "c_base": args.c_base,
            "eps": args.eps,
            "Lmin": args.Lmin,
            "Lmax": args.Lmax,
            "warmup_steps": args.warmup_steps,
            "eta_min": args.eta_min,
            "eta_max": args.eta_max,
            "theta_lr": args.theta_lr,
            "clip_grad": args.clip_grad,
            "val_meta_batches": args.val_meta_batches,
            "eta_change_ratio": args.eta_change_ratio,
            "wd": args.wd,
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=wandb_config,
        )

    # ------------------------------------------------------------------
    # Load CIFAR-10 and preprocess
    # ------------------------------------------------------------------
    (xtr, ytr), (xte, yte) = load_cifar10()

    # convert to float32 in [0, 1] to match preprocess stats
    xtr = np.asarray(xtr, dtype=np.float32) / 255.0
    xte = np.asarray(xte, dtype=np.float32) / 255.0
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm, _ = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)  # [3]
    std = np.array(norm["std"], np.float32)    # [3]
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train_raw, y_train = xtr[train_idx], ytr[train_idx]
    x_val_raw, y_val = xtr[val_idx], ytr[val_idx]
    x_test_raw, y_test = xte, yte

    x_train = to_nchw_and_norm(x_train_raw, mean, std)
    x_val = to_nchw_and_norm(x_val_raw, mean, std)
    x_test = to_nchw_and_norm(x_test_raw, mean, std)

    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)
    y_test_t = torch.from_numpy(y_test)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)

    cfg = LoaderCfg(batch_size=args.bs, num_workers=0)

    train_loader, val_loader = make_train_val_loaders(
        train_dataset,
        val_dataset,
        cfg,
        seed=args.seed,
        train_shuffle=True,
        val_shuffle=True,
        train_drop_last=True,
        val_drop_last=True,
    )

    def infinite_loader(loader):
        """Yield batches from a DataLoader forever."""
        while True:
            for batch in loader:
                yield batch

    val_iter = infinite_loader(val_loader)

    val_eval_loader = make_eval_loader(val_dataset, batch_size=512, num_workers=0, pin_memory=False)
    test_eval_loader = make_eval_loader(test_dataset, batch_size=512, num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------
    # Models and optimizers
    # ------------------------------------------------------------------
    net = Conv2DBaseline().to(device)
    learner = LearnedL(L_min=args.Lmin, L_max=args.Lmax, hidden=32).to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=args.theta_lr)
    ce = nn.CrossEntropyLoss()

    params = list(net.parameters())

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    curve_logger = BatchLossLogger(
        run_dir,
        meta={
            "method": "learned_l_f1_cifar10_conv2d_online_pt",
            "seed": args.seed,
            "opt": "learnedL",
            "lr": args.theta_lr,
        },
    )
    mech_path = run_dir / "mechanism.csv"
    eta_stats_path = run_dir / "eta_stats.csv"
    train_log_path = run_dir / "train_log.csv"
    time_log_path = run_dir / "time_log.csv"
    result_path = run_dir / "result.json"

    with open(mech_path, "w") as f:
        f.write("iter,epoch,eta_t,L_theta,phi_t\n")
    with open(eta_stats_path, "w") as f:
        f.write("epoch,eta_mean,eta_std,eta_min,eta_max,eta_p50,eta_p90,eta_p99,"
                "L_mean,phi_mean,dot_mean\n")
    with open(train_log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")

    with open(time_log_path, "w") as f:
        f.write("elapsed_sec,train_loss,train_acc,test_loss,test_acc\n")


    global_step = 0
    start_time = time.time()
    eta_prev = None  # for per-step eta change limiting

    # ------------------------------------------------------------------
    # Training loop (online meta-learning BF1-style)
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        curve_logger.on_epoch_begin(epoch)
        train_loss_sum = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0


        # epoch stats
        eta_hist = []
        L_hist = []
        phi_hist = []
        dot_hist = []

        net.train()
        learner.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # 1) train loss and gradients g_t
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)
            train_loss_sum += float(train_loss.item())
            train_batches += 1
            preds = logits_tr.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)


            grads = torch.autograd.grad(
                train_loss,
                params,
                create_graph=False,
                retain_graph=False,
            )
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            ]

            # phi_t = log ||g_t||
            g_norm_sq = sum((g.detach() ** 2).sum() for g in grads)
            g_norm = torch.sqrt(g_norm_sq + args.eps)
            phi = torch.log(g_norm + args.eps).view(1, 1)  # [1,1]
            phi_in = phi.to(device)

            # 2) meta signal: average over K val batches to reduce noise
            K = max(1, int(args.val_meta_batches))
            dot_sum = torch.zeros([], device=device, dtype=torch.float32)
            val_loss_sum = torch.zeros([], device=device, dtype=torch.float32)

            for _ in range(K):
                xv, yv = next(val_iter)
                xv = xv.to(device)
                yv = yv.to(device)

                logits_val = net(xv)
                val_loss_k = ce(logits_val, yv)
                val_loss_sum = val_loss_sum + val_loss_k.detach()

                grad_val = torch.autograd.grad(
                    val_loss_k,
                    params,
                    create_graph=False,
                    retain_graph=False,
                )
                grad_val = [
                    gv if gv is not None else torch.zeros_like(p)
                    for gv, p in zip(grad_val, params)
                ]

                dot_k = torch.zeros([], device=device, dtype=torch.float32)
                for gv, g in zip(grad_val, grads):
                    dot_k = dot_k + (gv.float() * g.detach().float()).sum()

                dot_sum = dot_sum + dot_k

            dot = dot_sum / float(K)
            val_loss = val_loss_sum / float(K)

            # 3) Online meta-update on theta
            L_theta = learner(phi_in)  # [1,1]
            eta = args.c_base / (L_theta + args.eps)

            if global_step < args.warmup_steps:
                warmup_max = min(
                    args.eta_max,
                    1.2 * args.c_base / (args.Lmin + args.eps),
                )
                eta = torch.clamp(eta, min=args.eta_min, max=warmup_max)
            else:
                eta = torch.clamp(eta, min=args.eta_min, max=args.eta_max)

            eta_scalar = eta.squeeze()

            # meta_loss = val_loss - eta * dot + small regularizer
            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(
                torch.square(torch.log(L_theta + args.eps))
            )

            theta_opt.zero_grad()
            meta_loss.backward()
            theta_opt.step()

            # 4) Clip g and update w with updated theta
            g_norm_for_clip = torch.sqrt(
                sum((g.detach() ** 2).sum() for g in grads) + 1e-12
            )
            if args.clip_grad is not None and args.clip_grad > 0.0:
                if g_norm_for_clip.item() > args.clip_grad:
                    clip_coef = args.clip_grad / float(g_norm_for_clip.item())
                else:
                    clip_coef = 1.0
            else:
                clip_coef = 1.0

            with torch.no_grad():
                L_now = learner(phi_in)
                eta_now = args.c_base / (L_now + args.eps)

                if global_step < args.warmup_steps:
                    warmup_max = min(
                        args.eta_max,
                        1.2 * args.c_base / (args.Lmin + args.eps),
                    )
                    eta_now = torch.clamp(eta_now, min=args.eta_min, max=warmup_max)
                else:
                    eta_now = torch.clamp(eta_now, min=args.eta_min, max=args.eta_max)

                eta_scalar_now = eta_now.squeeze()

                # --- per-step eta change limiter ---
                if eta_prev is None:
                    eta_limited = eta_scalar_now
                else:
                    r = float(args.eta_change_ratio)
                    if r is not None and r > 0.0:
                        lo = eta_prev * (1.0 - r)
                        hi = eta_prev * (1.0 + r)
                        eta_limited = torch.clamp(eta_scalar_now, min=lo, max=hi)
                    else:
                        eta_limited = eta_scalar_now

                eta_limited = torch.clamp(eta_limited, min=args.eta_min, max=args.eta_max)
                eta_prev = eta_limited.detach()

                for p, g in zip(params, grads):
                    g_update = g * clip_coef
                    if args.wd is not None and args.wd > 0.0:
                        g_update = g_update + args.wd * p.data
                    p.data -= eta_limited.to(p.device).to(p.dtype) * g_update

                # mechanism log
                with open(mech_path, "a") as f:
                    f.write(
                        f"{global_step},{epoch},"
                        f"{float(eta_limited.item()):.6g},"
                        f"{float(L_now.squeeze().item()):.6g},"
                        f"{float(phi.squeeze().item()):.6g}\n"
                    )

            # per-step stats
            eta_hist.append(float(eta_limited.item()))
            L_hist.append(float(L_now.squeeze().item()))
            phi_hist.append(float(phi.squeeze().item()))
            dot_hist.append(float(dot.detach().item()))

            curve_logger.on_train_batch_end(float(train_loss.item()))
            global_step += 1

        # ------------------------------------------------------------------
        # Epoch-level evaluation
        # ------------------------------------------------------------------
        train_loss_epoch = train_loss_sum / max(train_batches, 1)
        val_loss_epoch, val_acc = evaluate_on_loader(net, device, val_eval_loader, ce)
        test_loss_epoch, test_acc = evaluate_on_loader(net, device, test_eval_loader, ce)
        train_acc_epoch = train_correct / max(train_total, 1)


        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        with open(train_log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{test_loss_epoch:.8f},"
                f"{test_acc:.6f}\n"
            )
        with open(time_log_path, "a") as f:
            f.write(
                f"{total_elapsed:.3f},"
                f"{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{test_loss_epoch:.8f},{test_acc:.6f}\n")

        # --- epoch-level eta statistics ---
        eta_arr = np.asarray(eta_hist, dtype=np.float64) if eta_hist else np.asarray([np.nan])
        L_arr = np.asarray(L_hist, dtype=np.float64) if L_hist else np.asarray([np.nan])
        phi_arr = np.asarray(phi_hist, dtype=np.float64) if phi_hist else np.asarray([np.nan])
        dot_arr = np.asarray(dot_hist, dtype=np.float64) if dot_hist else np.asarray([np.nan])

        eta_mean = float(np.nanmean(eta_arr))
        eta_std = float(np.nanstd(eta_arr))
        eta_minv = float(np.nanmin(eta_arr))
        eta_maxv = float(np.nanmax(eta_arr))
        eta_p50 = float(np.nanpercentile(eta_arr, 50))
        eta_p90 = float(np.nanpercentile(eta_arr, 90))
        eta_p99 = float(np.nanpercentile(eta_arr, 99))
        L_mean = float(np.nanmean(L_arr))
        phi_mean = float(np.nanmean(phi_arr))
        dot_mean = float(np.nanmean(dot_arr))

        with open(eta_stats_path, "a") as f:
            f.write(
                f"{epoch},{eta_mean:.8g},{eta_std:.8g},{eta_minv:.8g},{eta_maxv:.8g},"
                f"{eta_p50:.8g},{eta_p90:.8g},{eta_p99:.8g},"
                f"{L_mean:.8g},{phi_mean:.8g},{dot_mean:.8g}\n"
            )

        print(
            f"[CIFAR10-Conv2D-F1-PT EPOCH {epoch}] "
            f"time={epoch_elapsed:.2f}s total={total_elapsed/60:.2f}min "
            f"train={train_loss_epoch:.4f} "
            f"val={val_loss_epoch:.4f} test={test_loss_epoch:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
            f"eta_mean={eta_mean:.3g} eta_p99={eta_p99:.3g}"
        )

        # WandB logging per epoch
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "val_loss": val_loss_epoch,
                    "test_loss": test_loss_epoch,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "time/epoch_sec": epoch_elapsed,
                    "time/total_sec": total_elapsed,
                    "eta/mean": eta_mean,
                    "eta/std": eta_std,
                    "eta/min": eta_minv,
                    "eta/max": eta_maxv,
                    "eta/p90": eta_p90,
                    "eta/p99": eta_p99,
                    "meta/dot_mean": dot_mean,
                }
            )

    curve_logger.on_train_end()
    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Final evaluation on full test set
    # ------------------------------------------------------------------
    net.eval()
    with torch.no_grad():
        logits_test = net(x_test_t.to(device))
        final_test_loss = ce(logits_test, y_test_t.to(device)).item()
        preds_test = logits_test.argmax(dim=1)
        final_test_acc = (preds_test == y_test_t.to(device)).float().mean().item()

    print(
        f"[RESULT-CIFAR10-Conv2D-F1-PT] "
        f"TestAcc={final_test_acc:.4f} TestLoss={final_test_loss:.4f} "
        f"(Total time={total_time/60:.2f} min)"
    )

    result = {
        "stage": "online-train",
        "dataset": "CIFAR-10 (Conv2D)",
        "method": "learned_l_f1_cifar10_conv2d_online_pt",
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "final_test_acc": float(final_test_acc),
        "final_test_loss": float(final_test_loss),
        "elapsed_sec": float(total_time),
        "run_dir": str(run_dir),
        "preprocess": str(Path(args.preprocess_dir) / f"seed_{args.data_seed}"),
        "hparams": {
            "c_base": args.c_base,
            "eps": args.eps,
            "Lmin": args.Lmin,
            "Lmax": args.Lmax,
            "warmup_steps": args.warmup_steps,
            "eta_min": args.eta_min,
            "eta_max": args.eta_max,
            "theta_lr": args.theta_lr,
            "clip_grad": args.clip_grad,
            "val_meta_batches": args.val_meta_batches,
            "eta_change_ratio": args.eta_change_ratio,
            "wd": args.wd,
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # WandB summary + finish
    if wandb_run is not None:
        wandb.run.summary["final_test_acc"] = float(final_test_acc)
        wandb.run.summary["final_test_loss"] = float(final_test_loss)
        wandb.run.summary["total_time_sec"] = float(total_time)
        # Optionally save log files
        for p in [
            curve_logger.curve_path,
            mech_path,
            eta_stats_path,
            train_log_path,
            time_log_path,
            result_path,
        ]:
            if Path(p).exists():
                wandb.save(str(p))
        wandb_run.finish()

            



if __name__ == "__main__":
    main()

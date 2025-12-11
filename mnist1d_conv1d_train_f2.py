#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Step-3 (B+F2) for MNIST-1D with a Conv1D backbone.

Step-3 (B+F2):
  - Learn an effective scalar L_theta from features
        phi_t = [log ||g_t||, log ||m_t||]
  - Step-size: eta_t = c_base / (L_theta(phi_t) + eps)
  - Single-step (T=1) online meta-learning on a *single task*,
    using momentum m_t as the search direction.

Key design:
  - Closely follows the F1 online meta-learning logic but with:
      * A momentum buffer m_t for each parameter:
            m_t = beta * m_{t-1} + (1 - beta) * g_t
      * Features: phi_t = [log ||g_t||, log ||m_t||]
      * Meta-loss uses the dot product between grad_val and momentum:
            dot = <grad_val, m_t>
  - For each train batch:
      1) Compute train loss and gradient g_t on the train batch
      2) Update momentum m_t
      3) Build phi_t from norms of g_t and m_t
      4) Sample a val batch, compute val_loss and grad_val
      5) Compute dot = <grad_val, m_t>
      6) Define meta_loss = val_loss - eta * dot + small regularizer
      7) Take one gradient step on theta (parameters of LearnedL)
      8) With the updated LearnedL, recompute eta and update w
         using the momentum direction (no SGD momentum optimizer)
  - No separate meta-train/meta-test split inside this script:
      net and LearnedL are trained jointly online on the same task.
  - Data preprocessing, normalization, and log file naming are aligned
    with the F1 PyTorch implementation.
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

from data_mnist1d import load_mnist1d

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ------------------------------- Utilities -------------------------------

class BatchLossLogger:
    """
    Lightweight BatchLossLogger aligned with the TF version.

    Writes CSV: curve_f2.csv with schema:
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
        self.curve_path = self.run_dir / "curve_f2.csv"
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


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(input_len: int = 40) -> nn.Module:
    """
    Conv1D backbone aligned with the TF version:

      - Conv1D(1 -> 32) + ReLU
      - MaxPool1d(2)
      - Conv1D(32 -> 64) + ReLU
      - GlobalAveragePool1d
      - Dense 64 -> 64 + ReLU
      - Dense 64 -> 10 (logits)
    """

    class Conv1DMNIST1D(nn.Module):
        def __init__(self, length: int):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.gap = nn.AdaptiveAvgPool1d(output_size=1)
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [N, 1, L]
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.gap(x)           # [N, 64, 1]
            x = x.view(x.size(0), -1) # [N, 64]
            x = self.fc1(x)
            x = self.relu(x)
            logits = self.fc2(x)      # [N, 10], logits
            return logits

    return Conv1DMNIST1D(input_len)


def load_artifacts(preprocess_dir: Path, seed: int):
    """
    Load preprocessing artifacts:

      - split.json: {train_idx, val_idx}
      - norm.json:  {mean, std}
      - meta.json:  (optional, only for bookkeeping)
    """
    pdir = Path(preprocess_dir) / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    meta_path = pdir / "meta.json"

    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"Preprocess artifacts not found under: {pdir}. Run preprocess first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return split, norm, meta


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-position standardization in NumPy: (x - mean) / std."""
    return (x - mean[None, :]) / std[None, :]


def to_NL(x: np.ndarray, length: int) -> np.ndarray:
    """
    Normalize input shape to [N, length]. Supports:
      - [N, length]
      - [N, length, 1]
      - [N, 1, length]
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(
        f"Unexpected x shape: {x.shape}, expected [N,{length}] or [N,{length},1]/[N,1,{length}]"
    )


# ------------------------- Learned L_theta (B+F2) -------------------------

class LearnedL(nn.Module):
    """
    MLP mapping phi = [log ||g||, log ||m||] to a positive scalar L_theta in [L_min, L_max].
    """

    def __init__(self, L_min: float = 1e-3, L_max: float = 1e3, hidden: int = 32):
        super().__init__()
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: [B, 2]
        return: L_theta in [L_min, L_max], shape [B, 1]
        """
        z = self.fc1(phi)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        s = self.sigmoid(self.fc_out(z))  # (0, 1)
        L_theta = self.L_min + (self.L_max - self.L_min) * s
        return L_theta


# ------------------------------ Main ------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--data_seed", type=int, default=42, help="fixed data split seed")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")

    # Learner hyperparameters (aligned with the F1 implementation)
    ap.add_argument("--c_base", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--Lmin", type=float, default=1e-3)
    ap.add_argument("--Lmax", type=float, default=1e3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--eta_max", type=float, default=1.0)
    ap.add_argument("--theta_lr", type=float, default=1e-3)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # Momentum hyperparameter
    ap.add_argument("--beta", type=float, default=0.9, help="momentum coefficient")

    # WandB logging
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        default="l2o-mnist1d",
        help="WandB project name",
    )
    ap.add_argument(
        "--wandb_group",
        type=str,
        default="mnist1d_conv1d_f2_bf_online",
        help="WandB group name",
    )
    ap.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="optional WandB run name, defaults to run_name",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    # Run directory: keep naming consistent with Conv1D BF F* scripts
    run_name = (
        f"mnist1d_conv1d_f2_data{args.data_seed}_seed{args.seed}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    # ---------------- WandB init (optional) ----------------
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError(
                "wandb is not installed, but --wandb was passed. "
                "Install via `pip install wandb` or disable --wandb."
            )
        wandb_config = {
            "stage": "online-train",
            "dataset": "MNIST-1D",
            "backbone": "Conv1D",
            "method": "learned_l_f2_mnist1d_conv1d_online_pt",
            "length": args.length,
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
            "beta": args.beta,
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=wandb_config,
        )

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=args.length, seed=args.data_seed)

    xtr = to_NL(xtr, length=args.length).astype(np.float32, copy=False)
    xte = to_NL(xte, length=args.length).astype(np.float32, copy=False)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm, _ = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)
    std = np.array(norm["std"], np.float32)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train_raw, y_train = xtr[train_idx], ytr[train_idx]
    x_val_raw, y_val = xtr[val_idx], ytr[val_idx]
    x_test_raw, y_test = xte, yte

    x_train = apply_norm_np(x_train_raw, mean, std)
    x_val = apply_norm_np(x_val_raw, mean, std)
    x_test = apply_norm_np(x_test_raw, mean, std)

    # Conv1D expects [N, 1, L]
    x_train_t = torch.from_numpy(x_train).unsqueeze(1)
    x_val_t = torch.from_numpy(x_val).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)
    y_test_t = torch.from_numpy(y_test)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
    )

    def infinite_loader(loader):
        """Yield batches from a DataLoader forever."""
        while True:
            for batch in loader:
                yield batch

    val_iter = infinite_loader(val_loader)

    val_eval_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_eval_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # ---------------- models ----------------
    net = build_model(args.length).to(device)
    learner = LearnedL(L_min=args.Lmin, L_max=args.Lmax).to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=args.theta_lr)
    ce = nn.CrossEntropyLoss()

    params = list(net.parameters())
    # Momentum buffers for each parameter (same shape, on the same device)
    momentum = [torch.zeros_like(p) for p in params]

    # ---------------- logs ----------------
    curve_logger = BatchLossLogger(
        run_dir,
        meta={
            "method": "learned_l_f2_mnist1d_conv1d",
            "seed": args.seed,
            "opt": "learnedL",
            "lr": args.theta_lr,
        },
    )
    mech_path = run_dir / "mechanism_f2.csv"
    train_log_path = run_dir / "train_log_f2.csv"
    time_log_path = run_dir / "time_log_f2.csv"
    result_path = run_dir / "result_f2.json"

    with open(mech_path, "w") as f:
        f.write("iter,epoch,eta_t,L_theta,log_g_norm,log_m_norm\n")
    with open(train_log_path, "w") as f:
        f.write("epoch,elapsed_sec,train_loss,val_loss,test_loss,val_acc,test_acc\n")
    with open(time_log_path, "w") as f:
        f.write("epoch,epoch_time_sec,total_elapsed_sec\n")

    global_step = 0
    start_time = time.time()

    # ---------------- training loop (online meta-learning) ----------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        curve_logger.on_epoch_begin(epoch)
        train_loss_sum = 0.0
        train_batches = 0

        net.train()
        learner.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # --- 1) Train batch: compute g_t ---
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)
            train_loss_sum += float(train_loss.item())
            train_batches += 1

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

            # --- 2) Update momentum m_t = beta * m_{t-1} + (1 - beta) * g_t ---
            with torch.no_grad():
                for m, g in zip(momentum, grads):
                    m.mul_(args.beta).add_(g, alpha=(1.0 - args.beta))

            # Compute norms for g_t and m_t
            g_norm_sq = sum((g.detach() ** 2).sum() for g in grads)
            g_norm = torch.sqrt(g_norm_sq + args.eps)

            m_norm_sq = sum((m.detach() ** 2).sum() for m in momentum)
            m_norm = torch.sqrt(m_norm_sq + args.eps)

            log_g_norm = torch.log(g_norm + args.eps)
            log_m_norm = torch.log(m_norm + args.eps)

            # phi_t = [log ||g_t||, log ||m_t||]
            phi = torch.stack([log_g_norm, log_m_norm], dim=0).view(1, 2)  # [1, 2]

            # --- 3) Val batch: compute val_loss and grad_val ---
            xv, yv = next(val_iter)
            xv = xv.to(device)
            yv = yv.to(device)

            logits_val = net(xv)
            val_loss = ce(logits_val, yv)
            grad_val = torch.autograd.grad(
                val_loss,
                params,
                create_graph=False,
                retain_graph=False,
            )
            grad_val = [
                gv if gv is not None else torch.zeros_like(p)
                for gv, p in zip(grad_val, params)
            ]

            # dot = sum <grad_val, stop_grad(m_t)>
            dot = torch.zeros([], device=device, dtype=torch.float32)
            for gv, m in zip(grad_val, momentum):
                dot = dot + (gv.float() * m.detach().float()).sum()

            # --- 4) Online meta-update on theta ---
            phi_in = phi.to(device)
            L_theta = learner(phi_in)  # [1, 1]
            eta = args.c_base / (L_theta + args.eps)

            # Warmup logic aligned with the F1 implementation
            if global_step < args.warmup_steps:
                warmup_max = min(
                    args.eta_max,
                    1.2 * args.c_base / (args.Lmin + args.eps),
                )
                eta = torch.clamp(eta, min=args.eta_min, max=warmup_max)
            else:
                eta = torch.clamp(eta, min=args.eta_min, max=args.eta_max)

            eta_scalar = eta.squeeze()  # scalar tensor

            # meta_loss = val_loss - eta * dot + small regularizer on log L_theta
            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(
                torch.square(torch.log(L_theta + args.eps))
            )

            theta_opt.zero_grad()
            meta_loss.backward()
            theta_opt.step()

            # --- 5) Clip momentum and update w using the *updated* theta ---
            # global norm for momentum (for clipping)
            m_norm_for_clip = torch.sqrt(
                sum((m.detach() ** 2).sum() for m in momentum) + 1e-12
            )
            if args.clip_grad is not None and args.clip_grad > 0.0:
                if m_norm_for_clip.item() > args.clip_grad:
                    clip_coef = args.clip_grad / float(m_norm_for_clip.item())
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

                for p, m in zip(params, momentum):
                    m_update = m * clip_coef
                    p.data -= eta_scalar_now.to(p.device).to(p.dtype) * m_update

                # Mechanism log
                with open(mech_path, "a") as f:
                    f.write(
                        f"{global_step},{epoch},"
                        f"{float(eta_scalar_now.item()):.6g},"
                        f"{float(L_now.squeeze().item()):.6g},"
                        f"{float(log_g_norm.item()):.6g},"
                        f"{float(log_m_norm.item()):.6g}\n"
                    )

            curve_logger.on_train_batch_end(float(train_loss.item()))
            global_step += 1

        # --- Epoch end evaluation (train loss, val/test loss + acc) ---
        train_loss_epoch = train_loss_sum / max(train_batches, 1)

        def eval_model(data_loader):
            net.eval()
            losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for xb_eval, yb_eval in data_loader:
                    xb_eval = xb_eval.to(device)
                    yb_eval = yb_eval.to(device)
                    logits_eval = net(xb_eval)
                    loss_eval = ce(logits_eval, yb_eval)
                    losses.append(float(loss_eval.item()))
                    preds = logits_eval.argmax(dim=1)
                    correct += (preds == yb_eval).sum().item()
                    total += yb_eval.size(0)
            return (
                np.mean(losses) if losses else float("nan"),
                correct / max(total, 1),
            )

        val_loss_epoch, val_acc = eval_model(val_eval_loader)
        test_loss_epoch, test_acc = eval_model(test_eval_loader)

        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        with open(train_log_path, "a") as f:
            f.write(
                f"{epoch},{total_elapsed:.3f},"
                f"{train_loss_epoch:.8f},"
                f"{val_loss_epoch:.8f},"
                f"{test_loss_epoch:.8f},"
                f"{val_acc:.6f},"
                f"{test_acc:.6f}\n"
            )
        with open(time_log_path, "a") as f:
            f.write(f"{epoch},{epoch_elapsed:.3f},{total_elapsed:.3f}\n")

        print(
            f"[MNIST1D-Conv1D-F2-PT EPOCH {epoch}] "
            f"time={epoch_elapsed:.2f}s total={total_elapsed/60:.2f}min "
            f"train={train_loss_epoch:.4f} "
            f"val={val_loss_epoch:.4f} test={test_loss_epoch:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

        # WandB per-epoch logging
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
                }
            )

    curve_logger.on_train_end()
    total_time = time.time() - start_time

    # ---------------- final eval on full test set ----------------
    net.eval()
    with torch.no_grad():
        logits_test = net(x_test_t.to(device))
        final_test_loss = ce(logits_test, y_test_t.to(device)).item()
        preds_test = logits_test.argmax(dim=1)
        final_test_acc = (preds_test == y_test_t.to(device)).float().mean().item()

    print(
        f"[RESULT-MNIST1D-Conv1D-F2-PT] TestAcc={final_test_acc:.4f} "
        f"TestLoss={final_test_loss:.4f} "
        f"(Total time={total_time/60:.2f} min)"
    )

    result = {
        "dataset": f"MNIST-1D(len={args.length})",
        "backbone": "Conv1D",
        "method": "learned_l_f2_mnist1d_conv1d_online_pt",
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "test_acc": float(final_test_acc),
        "test_loss": float(final_test_loss),
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
            "beta": args.beta,
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # WandB summary + upload logs
    if wandb_run is not None:
        wandb_run.summary["final_test_acc"] = float(final_test_acc)
        wandb_run.summary["final_test_loss"] = float(final_test_loss)
        wandb_run.summary["total_time_sec"] = float(total_time)

        for p in [
            curve_logger.curve_path,
            mech_path,
            train_log_path,
            time_log_path,
            result_path,
        ]:
            if Path(p).exists():
                wandb.save(str(p))

        wandb_run.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch BF-style (B+F1) online meta-learning on CIFAR-10 with a ResNet-18 backbone.
-----------------------------------------------------------------------------------

This script is the CIFAR-10 ResNet-18 counterpart of the MNIST-1D Conv1D F1 BF-style code.

Step-3 (B+F1):
  - Learn an effective scalar L_theta from feature phi_t = log ||g_t||
  - Step-size: eta_t = c_base / (L_theta(phi_t) + eps)
  - Single-step (T=1) online meta-learning on a *single task* (CIFAR-10).

Key design:
  - For each train batch:
      1) Compute train loss and gradient g_t on the train batch
      2) Sample a val batch, compute val_loss and grad_val
      3) Compute dot = <grad_val, g_t>
      4) Define meta_loss = val_loss - eta * dot + small regularizer
      5) Take one gradient step on theta (parameters of LearnedL)
      6) With the updated LearnedL, recompute eta and update w
         using a plain SGD-like step (no momentum)
  - No separate meta-train/meta-test split:
      ResNet-18 and LearnedL are trained jointly online on the same CIFAR-10 task.
  - Data preprocessing, normalization, and log file naming are aligned
    with the Conv2D CIFAR-10 F1 implementation, using artifacts under
    preprocess_dir (e.g. artifacts/cifar10_conv2d_preprocess/seed_{data_seed}).
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

from data_cifar10 import load_cifar10  # as in your project

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ------------------------------- Utilities -------------------------------

class BatchLossLogger:
    """
    Lightweight BatchLossLogger aligned with MNIST-1D / CIFAR-10 BF F1 code.

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


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_artifacts(preprocess_dir: Path, seed: int):
    """
    Load preprocessing artifacts for CIFAR-10 Conv2D/ResNet:

      - split.json: {train_idx, val_idx}
      - norm.json:  {mean, std} (per-channel stats on train subset, x in [0,1])
      - meta.json:  (optional, for bookkeeping)
    """
    pdir = Path(preprocess_dir) / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    meta_path = pdir / "meta.json"

    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"Preprocess artifacts not found under: {pdir}. Run CIFAR-10 preprocess first."
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


def to_nchw_and_norm(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert CIFAR-10 images to [N, C, H, W] and apply per-channel normalization.

    Assumptions:
      - x has shape [N, 32, 32, 3] or [N, 3, 32, 32]
      - mean, std are 1D per-channel vectors (len=3) computed on x in [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 4:
        raise AssertionError(
            f"Expected CIFAR-10 images with 4 dimensions, got shape {x.shape}"
        )

    # Ensure NCHW layout
    if x.shape[1] == 3:
        # Already [N, C, H, W]
        x_nchw = x
    elif x.shape[-1] == 3:
        # [N, H, W, C] -> [N, C, H, W]
        x_nchw = np.transpose(x, (0, 3, 1, 2))
    else:
        raise AssertionError(
            f"Unexpected CIFAR-10 shape {x.shape}, cannot infer channel dimension."
        )

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # Reshape per-channel stats to [1, C, 1, 1]
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


# --------------------------- ResNet-18 backbone ---------------------------

class BasicBlock(nn.Module):
    """
    Standard BasicBlock for ResNet-18, adapted to CIFAR-10:
      - No downsampling in the first block of each stage unless stride != 1 or channel change.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 variant for CIFAR-10 (32x32 inputs):

      - First conv: 3x3, stride=1, no maxpool
      - 4 stages: [2,2,2,2] BasicBlocks with channels [64,128,256,512]
      - Global average pooling and a linear 512->10 classifier
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)  # 4x4

        x = self.avgpool(x)           # [N, 512, 1, 1]
        x = torch.flatten(x, 1)       # [N, 512]
        logits = self.fc(x)           # [N, 10]
        return logits


def build_model() -> nn.Module:
    """Build a ResNet-18 backbone for CIFAR-10."""
    return ResNet18CIFAR(num_classes=10)


# ------------------------- Learned L_theta (B+F1) -------------------------

class LearnedL(nn.Module):
    """
    Simple MLP mapping phi = log ||g|| to a positive scalar L_theta in [L_min, L_max].
    """

    def __init__(self, L_min: float = 1e-3, L_max: float = 1e3, hidden: int = 32):
        super().__init__()
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: [B, 1]
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
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--data_seed", type=int, default=42, help="fixed data split seed")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/cifar10_conv2d_preprocess",
        help="directory containing CIFAR-10 split.json / norm.json",
    )

    # Learner / optimizer hyperparameters (aligned with Conv2D F1 version)
    ap.add_argument("--c_base", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--Lmin", type=float, default=1e-3)
    ap.add_argument("--Lmax", type=float, default=1e3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--eta_max", type=float, default=1.0)
    ap.add_argument("--theta_lr", type=float, default=1e-3)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # WandB logging
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        default="l2o-cifar10",
        help="WandB project name",
    )
    ap.add_argument(
        "--wandb_group",
        type=str,
        default="cifar10_resnet18_f1_bf1pt",
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

    # Run directory: encode CIFAR-10 + ResNet18 + F1
    run_name = (
        f"cifar10_resnet18_f1_data{args.data_seed}_seed{args.seed}_"
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
            "dataset": "CIFAR-10",
            "backbone": "ResNet18",
            "method": "learned_l_f1_cifar10_resnet18_online_pt",
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
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=wandb_config,
        )

    # ---------------- data ----------------
    # CIFAR-10 loader does not take a `seed` argument.
    # The train/val split is controlled by split.json under preprocess_dir.
    (xtr, ytr), (xte, yte) = load_cifar10()

    # Cast to float32 in [0, 1] to match how mean/std were computed in preprocess.
    xtr = np.asarray(xtr, dtype=np.float32) / 255.0   # [N, 32, 32, 3] in [0, 1]
    xte = np.asarray(xte, dtype=np.float32) / 255.0
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm, _ = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)  # shape [3]
    std = np.array(norm["std"], np.float32)    # shape [3]
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train_raw, y_train = xtr[train_idx], ytr[train_idx]
    x_val_raw, y_val = xtr[val_idx], ytr[val_idx]
    x_test_raw, y_test = xte, yte

    x_train = to_nchw_and_norm(x_train_raw, mean, std)
    x_val = to_nchw_and_norm(x_val_raw, mean, std)
    x_test = to_nchw_and_norm(x_test_raw, mean, std)

    # ResNet expects [N, C, H, W]
    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)
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
    net = build_model().to(device)
    learner = LearnedL(L_min=args.Lmin, L_max=args.Lmax).to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=args.theta_lr)
    ce = nn.CrossEntropyLoss()

    params = list(net.parameters())

    # ---------------- logs ----------------
    curve_logger = BatchLossLogger(
        run_dir,
        meta={
            "method": "learned_l_f1_cifar10_resnet18",
            "seed": args.seed,
            "opt": "learnedL",
            "lr": args.theta_lr,
        },
    )
    mech_path = run_dir / "mechanism_f1.csv"
    train_log_path = run_dir / "train_log_f1.csv"
    time_log_path = run_dir / "time_log_f1.csv"
    result_path = run_dir / "result_f1.json"

    with open(mech_path, "w") as f:
        f.write("iter,epoch,eta_t,L_theta,phi_t\n")
    with open(train_log_path, "w") as f:
        f.write("epoch,elapsed_sec,train_loss,val_loss,test_loss,val_acc,test_acc\n")
    with open(time_log_path, "w") as f:
        f.write("epoch,epoch_time_sec,total_elapsed_sec\n")

    global_step = 0
    start_time = time.time()

    # ---------------- training loop (online meta-learning, F1) ----------------
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

            # phi_t = log ||g_t|| (unclipped)
            g_norm_sq = sum((g.detach() ** 2).sum() for g in grads)
            g_norm = torch.sqrt(g_norm_sq + args.eps)
            phi = torch.log(g_norm + args.eps).view(1, 1)  # [1, 1]

            # --- 2) Val batch: compute val_loss and grad_val ---
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

            # dot = sum <grad_val, stop_grad(g)>
            dot = torch.zeros([], device=device, dtype=torch.float32)
            for gv, g in zip(grad_val, grads):
                dot = dot + (gv.float() * g.detach().float()).sum()

            # --- 3) Online meta-update on theta (F1 feature) ---
            phi_in = phi.to(device)
            L_theta = learner(phi_in)  # [1, 1]
            eta = args.c_base / (L_theta + args.eps)

            if global_step < args.warmup_steps:
                # Warmup logic aligned with other BF implementations
                warmup_max = min(
                    args.eta_max,
                    1.2 * args.c_base / (args.Lmin + args.eps),
                )
                eta = torch.clamp(eta, min=args.eta_min, max=warmup_max)
            else:
                eta = torch.clamp(eta, min=args.eta_min, max=args.eta_max)

            eta_scalar = eta.squeeze()  # scalar tensor

            # meta_loss = val_loss - eta * dot + small regularizer on L_theta
            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(
                torch.square(torch.log(L_theta + args.eps))
            )

            theta_opt.zero_grad()
            meta_loss.backward()
            theta_opt.step()

            # --- 4) Clip g and update w using the *updated* theta ---
            # global norm for grads (for clipping)
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
                    eta_now = torch.clamp(
                        eta_now, min=args.eta_min, max=warmup_max
                    )
                else:
                    eta_now = torch.clamp(
                        eta_now, min=args.eta_min, max=args.eta_max
                    )
                eta_scalar_now = eta_now.squeeze()

                for p, g in zip(params, grads):
                    g_update = g * clip_coef
                    p.data -= eta_scalar_now.to(p.device).to(p.dtype) * g_update

                # Mechanism log
                with open(mech_path, "a") as f:
                    f.write(
                        f"{global_step},{epoch},"
                        f"{float(eta_scalar_now.item()):.6g},"
                        f"{float(L_now.squeeze().item()):.6g},"
                        f"{float(phi.squeeze().item()):.6g}\n"
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
            f"[CIFAR10-ResNet18-F1-PT EPOCH {epoch}] "
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
        f"[RESULT-CIFAR10-ResNet18-F1-PT] TestAcc={final_test_acc:.4f} "
        f"TestLoss={final_test_loss:.4f} "
        f"(Total time={total_time/60:.2f} min)"
    )

    result = {
        "dataset": "CIFAR-10(32x32x3)",
        "backbone": "ResNet18",
        "method": "learned_l_f1_cifar10_resnet18_online_pt",
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

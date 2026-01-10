#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch BF-style (B+F3) online meta-learning on CIFAR-100 with a ResNet-50 backbone.
-----------------------------------------------------------------------------------

Stabilized / aligned version:
  - Average meta dot/val_loss over multiple val batches (--val_meta_batches)
  - Limit per-step eta relative change (--eta_change_ratio)
  - Optional weight decay in the manual w update (--wd)
  - Per-epoch eta/L/phi/dot stats logged to eta_stats.csv

F3 feature:
  phi = [log||g||, log||m||, ||m-g||]
Meta dot direction: <grad_val, m>  (momentum direction)
Update direction: m (EMA gradient)

Data pipeline:
  - Uses preprocess artifacts (split.json / norm.json / meta.json) produced by
    cifar100_resnet50_preprocess.py (Strategy B: train-subset mean/std).
  - Converts images to NCHW and applies (x - mean) / std offline into TensorDataset.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from data_cifar100 import load_cifar100
from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ------------------------------- Utilities -------------------------------

class BatchLossLogger:
    """
    Writes CSV: curve_f3.csv with schema:
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
        self.curve_path = self.run_dir / "curve_f3.csv"
        with open(self.curve_path, "w", encoding="utf-8") as f:
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
        with open(self.curve_path, "a", encoding="utf-8") as f:
            for r in self.rows:
                f.write(",".join(map(str, r)) + "\n")
        self.rows.clear()


def set_seed(s: int):
    np.random.seed(int(s))
    torch.manual_seed(int(s))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(s))


def load_artifacts(preprocess_dir: Path, seed: int):
    pdir = Path(preprocess_dir) / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    meta_path = pdir / "meta.json"

    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Preprocess artifacts not found under: {pdir}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return split, norm, meta


def to_nchw_and_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Accepts x in either NHWC or NCHW and returns NCHW normalized.
    Expects float32 in [0,1] already (caller responsibility).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise AssertionError(f"Expected images with 4 dimensions, got shape {x.shape}")

    if x.shape[1] == 3:
        x_nchw = x
    elif x.shape[-1] == 3:
        x_nchw = np.transpose(x, (0, 3, 1, 2))
    else:
        raise AssertionError(f"Unexpected image shape {x.shape}, cannot infer channel dimension.")

    mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x_nchw - mean) / std


def evaluate_on_loader(model: nn.Module, device: torch.device, loader, criterion: nn.Module):
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


# --------------------------- ResNet-50 backbone ---------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        out_planes = planes * self.expansion

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet50CIFAR(nn.Module):
    """
    CIFAR-style stem:
      - 3x3 conv, stride=1, padding=1
      - no maxpool
    Standard stage strides:
      layer1 stride=1, layer2 stride=2, layer3 stride=2, layer4 stride=2
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [Bottleneck(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(num_classes: int = 100) -> nn.Module:
    return ResNet50CIFAR(num_classes=num_classes)


# ------------------------- Learned L_theta (B+F3) -------------------------

class LearnedL(nn.Module):
    """
    phi = [log||g||, log||m||, ||m-g||] -> L_theta in [Lmin, Lmax]
    """
    def __init__(self, L_min: float = 1e-3, L_max: float = 1e3, hidden: int = 32):
        super().__init__()
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.fc1(phi))
        z = self.relu(self.fc2(z))
        s = self.sigmoid(self.fc_out(z))
        return self.L_min + (self.L_max - self.L_min) * s


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)

    # IMPORTANT: preprocess artifacts from cifar100_resnet50_preprocess.py
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/cifar100_resnet50_preprocess")

    # core knobs
    ap.add_argument("--c_base", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--Lmin", type=float, default=1e-3)
    ap.add_argument("--Lmax", type=float, default=1e3)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--eta_max", type=float, default=1.0)
    ap.add_argument("--theta_lr", type=float, default=1e-3)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.9, help="EMA coefficient for gradient moving average m_t")

    # aligned stability / regularization knobs
    ap.add_argument("--val_meta_batches", type=int, default=2,
                    help="number of val mini-batches to average for meta dot (reduces noise)")
    ap.add_argument("--eta_change_ratio", type=float, default=0.05,
                    help="max relative change of eta per step, e.g. 0.05 means +/-5%")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay applied in the manual w update (0 disables)")

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-cifar100")
    ap.add_argument("--wandb_group", type=str, default="cifar100_resnet50_f3_bf3pt")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    run_name = (
        f"cifar100_resnet50_f3_data{args.data_seed}_seed{args.seed}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    wandb_run = None
    method_name = "learned_l_f3_cifar100_resnet50_online_pt"

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config={
                "stage": "online-train",
                "dataset": "CIFAR-100",
                "backbone": "ResNet50(CIFAR-stem)",
                "method": method_name,
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
                "val_meta_batches": args.val_meta_batches,
                "eta_change_ratio": args.eta_change_ratio,
                "wd": args.wd,
            },
        )

    # ------------------------------ data ------------------------------
    (xtr, ytr), (xte, yte) = load_cifar100()
    xtr = np.asarray(xtr, dtype=np.float32) / 255.0
    xte = np.asarray(xte, dtype=np.float32) / 255.0
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm, _ = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)
    std = np.array(norm["std"], np.float32)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train = to_nchw_and_norm(xtr[train_idx], mean, std)
    y_train = ytr[train_idx]
    x_val = to_nchw_and_norm(xtr[val_idx], mean, std)
    y_val = ytr[val_idx]
    x_test = to_nchw_and_norm(xte, mean, std)
    y_test = yte

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    x_test_t = torch.from_numpy(x_test)
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
        while True:
            for batch in loader:
                yield batch

    val_iter = infinite_loader(val_loader)
    val_eval_loader = make_eval_loader(val_dataset, batch_size=512, num_workers=0, pin_memory=False)
    test_eval_loader = make_eval_loader(test_dataset, batch_size=512, num_workers=0, pin_memory=False)

    # ------------------------------ models ------------------------------
    net = build_model(num_classes=100).to(device)
    learner = LearnedL(L_min=args.Lmin, L_max=args.Lmax, hidden=32).to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=args.theta_lr)
    ce = nn.CrossEntropyLoss()

    params = list(net.parameters())
    m_buffers = [torch.zeros_like(p, device=device) for p in params]

    # ------------------------------ logs ------------------------------
    curve_logger = BatchLossLogger(
        run_dir,
        meta={"method": method_name, "seed": args.seed, "opt": "learnedL", "lr": args.theta_lr},
    )
    mech_path = run_dir / "mechanism.csv"
    eta_stats_path = run_dir / "eta_stats.csv"
    train_log_path = run_dir / "train_log.csv"
    time_log_path = run_dir / "time_log.csv"
    result_path = run_dir / "result.json"

    with open(mech_path, "w", encoding="utf-8") as f:
        f.write("iter,epoch,eta_t,L_theta,phi_g,phi_m,phi_diff\n")
    with open(eta_stats_path, "w", encoding="utf-8") as f:
        f.write("epoch,eta_mean,eta_std,eta_min,eta_max,eta_p50,eta_p90,eta_p99,"
                "L_mean,phi_g_mean,phi_m_mean,phi_diff_mean,dot_mean\n")

    # IMPORTANT: include val_loss/val_acc in the tables (first version already includes them)
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n")
    with open(time_log_path, "w", encoding="utf-8") as f:
        f.write("elapsed_sec,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n")

    global_step = 0
    start_time = time.time()
    eta_prev = None

    # ------------------------------ training loop ------------------------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        curve_logger.on_epoch_begin(epoch)

        train_loss_sum = 0.0
        train_batches = 0
        train_correct = 0
        train_total = 0

        eta_hist, L_hist = [], []
        phi_g_hist, phi_m_hist, phi_diff_hist = [], [], []
        dot_hist = []

        net.train()
        learner.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # train grad
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)
            train_loss_sum += float(train_loss.item())
            train_batches += 1

            preds = logits_tr.argmax(dim=1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(yb.size(0))

            grads = torch.autograd.grad(train_loss, params, create_graph=False, retain_graph=False)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # EMA update (in-place) for momentum direction m
            with torch.no_grad():
                for m, g in zip(m_buffers, grads):
                    m.mul_(args.beta).add_(g, alpha=(1.0 - args.beta))

            # F3 feature
            g_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads) + args.eps)
            m_norm = torch.sqrt(sum((m ** 2).sum() for m in m_buffers) + args.eps)
            diff_norm = torch.sqrt(sum(((m - g.detach()) ** 2).sum() for m, g in zip(m_buffers, grads)) + args.eps)

            phi_g = torch.log(g_norm + args.eps)
            phi_m = torch.log(m_norm + args.eps)
            phi_diff = diff_norm

            phi = torch.stack([phi_g, phi_m, phi_diff], dim=0).view(1, 3)
            phi_in = phi.to(device)

            # meta signal avg over K val batches
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

                grad_val = torch.autograd.grad(val_loss_k, params, create_graph=False, retain_graph=False)
                grad_val = [gv if gv is not None else torch.zeros_like(p) for gv, p in zip(grad_val, params)]

                # dot along momentum direction m
                dot_k = torch.zeros([], device=device, dtype=torch.float32)
                for gv, m in zip(grad_val, m_buffers):
                    dot_k = dot_k + (gv.float() * m.float()).sum()
                dot_sum = dot_sum + dot_k

            dot = dot_sum / float(K)
            val_loss = val_loss_sum / float(K)

            # theta update
            L_theta = learner(phi_in)
            eta = args.c_base / (L_theta + args.eps)

            if global_step < args.warmup_steps:
                warmup_max = min(args.eta_max, 1.2 * args.c_base / (args.Lmin + args.eps))
                eta = torch.clamp(eta, min=args.eta_min, max=warmup_max)
            else:
                eta = torch.clamp(eta, min=args.eta_min, max=args.eta_max)

            eta_scalar = eta.squeeze()

            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(torch.square(torch.log(L_theta + args.eps)))

            theta_opt.zero_grad()
            meta_loss.backward()
            theta_opt.step()

            # clip momentum and update w with updated theta + eta limiter + wd
            m_norm_for_clip = torch.sqrt(sum((m ** 2).sum() for m in m_buffers) + 1e-12)
            if args.clip_grad is not None and args.clip_grad > 0.0:
                clip_coef = args.clip_grad / float(m_norm_for_clip.item()) if m_norm_for_clip.item() > args.clip_grad else 1.0
            else:
                clip_coef = 1.0

            with torch.no_grad():
                L_now = learner(phi_in)
                eta_now = args.c_base / (L_now + args.eps)

                if global_step < args.warmup_steps:
                    warmup_max = min(args.eta_max, 1.2 * args.c_base / (args.Lmin + args.eps))
                    eta_now = torch.clamp(eta_now, min=args.eta_min, max=warmup_max)
                else:
                    eta_now = torch.clamp(eta_now, min=args.eta_min, max=args.eta_max)

                eta_scalar_now = eta_now.squeeze()

                # eta limiter
                if eta_prev is None:
                    eta_limited = eta_scalar_now
                else:
                    r = float(args.eta_change_ratio)
                    if r > 0.0:
                        lo = eta_prev * (1.0 - r)
                        hi = eta_prev * (1.0 + r)
                        eta_limited = torch.clamp(eta_scalar_now, min=lo, max=hi)
                    else:
                        eta_limited = eta_scalar_now

                eta_limited = torch.clamp(eta_limited, min=args.eta_min, max=args.eta_max)
                eta_prev = eta_limited.detach()

                for p, m in zip(params, m_buffers):
                    m_update = (m * clip_coef)
                    if args.wd is not None and args.wd > 0.0:
                        m_update = m_update + args.wd * p.data
                    p.data -= eta_limited.to(p.device).to(p.dtype) * m_update

                with open(mech_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{global_step},{epoch},"
                        f"{float(eta_limited.item()):.6g},"
                        f"{float(L_now.squeeze().item()):.6g},"
                        f"{float(phi_g.item()):.6g},"
                        f"{float(phi_m.item()):.6g},"
                        f"{float(phi_diff.item()):.6g}\n"
                    )

            # stats
            eta_hist.append(float(eta_limited.item()))
            L_hist.append(float(L_now.squeeze().item()))
            phi_g_hist.append(float(phi_g.item()))
            phi_m_hist.append(float(phi_m.item()))
            phi_diff_hist.append(float(phi_diff.item()))
            dot_hist.append(float(dot.detach().item()))

            curve_logger.on_train_batch_end(float(train_loss.item()))
            global_step += 1

        # epoch eval
        train_loss_epoch = train_loss_sum / max(train_batches, 1)
        train_acc_epoch = train_correct / max(train_total, 1)
        val_loss_epoch, val_acc = evaluate_on_loader(net, device, val_eval_loader, ce)
        test_loss_epoch, test_acc = evaluate_on_loader(net, device, test_eval_loader, ce)

        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{val_loss_epoch:.8f},{val_acc:.6f},"
                f"{test_loss_epoch:.8f},{test_acc:.6f}\n"
            )
        with open(time_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{total_elapsed:.3f},"
                f"{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{val_loss_epoch:.8f},{val_acc:.6f},"
                f"{test_loss_epoch:.8f},{test_acc:.6f}\n"
            )

        # eta stats
        eta_arr = np.asarray(eta_hist, dtype=np.float64) if eta_hist else np.asarray([np.nan])
        L_arr = np.asarray(L_hist, dtype=np.float64) if L_hist else np.asarray([np.nan])
        pg_arr = np.asarray(phi_g_hist, dtype=np.float64) if phi_g_hist else np.asarray([np.nan])
        pm_arr = np.asarray(phi_m_hist, dtype=np.float64) if phi_m_hist else np.asarray([np.nan])
        pd_arr = np.asarray(phi_diff_hist, dtype=np.float64) if phi_diff_hist else np.asarray([np.nan])
        dot_arr = np.asarray(dot_hist, dtype=np.float64) if dot_hist else np.asarray([np.nan])

        eta_mean = float(np.nanmean(eta_arr))
        eta_std = float(np.nanstd(eta_arr))
        eta_minv = float(np.nanmin(eta_arr))
        eta_maxv = float(np.nanmax(eta_arr))
        eta_p50 = float(np.nanpercentile(eta_arr, 50))
        eta_p90 = float(np.nanpercentile(eta_arr, 90))
        eta_p99 = float(np.nanpercentile(eta_arr, 99))
        L_mean = float(np.nanmean(L_arr))
        pg_mean = float(np.nanmean(pg_arr))
        pm_mean = float(np.nanmean(pm_arr))
        pd_mean = float(np.nanmean(pd_arr))
        dot_mean = float(np.nanmean(dot_arr))

        with open(eta_stats_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{eta_mean:.8g},{eta_std:.8g},{eta_minv:.8g},{eta_maxv:.8g},"
                f"{eta_p50:.8g},{eta_p90:.8g},{eta_p99:.8g},"
                f"{L_mean:.8g},{pg_mean:.8g},{pm_mean:.8g},{pd_mean:.8g},{dot_mean:.8g}\n"
            )

        print(
            f"[CIFAR100-ResNet50-F1-PT EPOCH {epoch}] "
            f"time={epoch_elapsed:.2f}s total={total_elapsed/60:.2f}min "
            f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} "
            f"val_loss={val_loss_epoch:.4f} val_acc={val_acc:.4f} "
            f"test_loss={test_loss_epoch:.4f} test_acc={test_acc:.4f} "
            f"eta_mean={eta_mean:.3g} eta_p99={eta_p99:.3g}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "val_loss": val_loss_epoch,
                    "test_loss": test_loss_epoch,
                    "train_acc": train_acc_epoch,
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

    # final eval
    # ---------------- final eval (SAFE: batched) ----------------
    if device.type == "cuda":
        torch.cuda.empty_cache()   

    final_test_loss, final_test_acc = evaluate_on_loader(net, device, test_eval_loader, ce)


    print(
        f"[RESULT-CIFAR100-ResNet50-F3-PT] TestAcc={final_test_acc:.4f} "
        f"TestLoss={final_test_loss:.4f} (Total time={total_time/60:.2f} min)"
    )

    result = {
        "dataset": "CIFAR-100(32x32x3)",
        "backbone": "ResNet50(CIFAR-stem)",
        "method": method_name,
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
            "beta": args.beta,
            "val_meta_batches": args.val_meta_batches,
            "eta_change_ratio": args.eta_change_ratio,
            "wd": args.wd,
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb.run.summary["final_test_acc"] = float(final_test_acc)
        wandb.run.summary["final_test_loss"] = float(final_test_loss)
        wandb.run.summary["total_time_sec"] = float(total_time)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare optimizers on MNIST-1D (Conv1D backbone) and log per-epoch metrics to W&B.

Methods:
  - sgd / adam / rmsprop: standard torch.optim
  - f1 / f2 / f3 / f3alpha: your online meta-learning step-3 logic (B+F*)

Per-epoch outputs:
  - train_loss, train_acc
  - test_loss,  test_acc

Data:
  - Loads normalized splits from artifacts/preprocess/seed_{data_seed}/
    Prefer exported NPZ (x_train/x_val/x_test), else fallback to split.json+norm.json.

Backbone:
  - Conv1D as in your F1/F2/F3 scripts (expects input [N,1,40]).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

from mnist1d_core import set_seed, load_mnist1d_from_artifacts


# -------------------------
# Models
# -------------------------
class Conv1DMNIST1D(nn.Module):
    """
    Conv1D backbone aligned with your F1/F2/F3 scripts:

      - Conv1D(1 -> 32) + ReLU
      - MaxPool1d(2)
      - Conv1D(32 -> 64) + ReLU
      - GlobalAveragePool1d
      - Dense 64 -> 64 + ReLU
      - Dense 64 -> 10 (logits)
    """
    def __init__(self, length: int = 40, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, L]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.gap(x)                # [N, 64, 1]
        x = x.view(x.size(0), -1)      # [N, 64]
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)             # logits


class LearnedL_F1(nn.Module):
    """phi = [log||g||] -> L_theta in [Lmin, Lmax]."""
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
        z = self.relu(self.fc1(phi))
        z = self.relu(self.fc2(z))
        s = self.sigmoid(self.fc_out(z))
        return self.L_min + (self.L_max - self.L_min) * s


class LearnedL_F2(nn.Module):
    """phi = [log||g||, log||m||] -> L_theta in [Lmin, Lmax]."""
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
        z = self.relu(self.fc1(phi))
        z = self.relu(self.fc2(z))
        s = self.sigmoid(self.fc_out(z))
        return self.L_min + (self.L_max - self.L_min) * s


class LearnedL_F3(nn.Module):
    """phi = [log||g||, log||m||, ||m-g||] -> L_theta in [Lmin, Lmax]."""
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


# -------------------------
# Data
# -------------------------
@dataclass
class DataSplits:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_splits(
    preprocess_dir: str,
    data_seed: int,
    as_conv1d: bool = True,
) -> DataSplits:
    arrays = load_mnist1d_from_artifacts(preprocess_dir=preprocess_dir, seed=data_seed)

    x_tr, y_tr = arrays["train"]
    x_va, y_va = arrays["val"]
    x_te, y_te = arrays["test"]

    # to torch
    x_tr_t = torch.from_numpy(np.asarray(x_tr, dtype=np.float32))
    x_va_t = torch.from_numpy(np.asarray(x_va, dtype=np.float32))
    x_te_t = torch.from_numpy(np.asarray(x_te, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))
    y_va_t = torch.from_numpy(np.asarray(y_va, dtype=np.int64))
    y_te_t = torch.from_numpy(np.asarray(y_te, dtype=np.int64))

    if as_conv1d:
        # [N,40] -> [N,1,40]
        x_tr_t = x_tr_t.unsqueeze(1)
        x_va_t = x_va_t.unsqueeze(1)
        x_te_t = x_te_t.unsqueeze(1)

    return DataSplits(
        x_train=x_tr_t, y_train=y_tr_t,
        x_val=x_va_t, y_val=y_va_t,
        x_test=x_te_t, y_test=y_te_t,
    )


def make_loaders(
    splits: DataSplits,
    bs: int,
    seed: int,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Returns:
      train_loader (shuffle True, drop_last True by default)
      val_loader   (shuffle True,  drop_last True)  # for F* val_iter
      train_eval_loader (shuffle False)
      test_eval_loader  (shuffle False)
    """
    g_train = torch.Generator()
    g_train.manual_seed(seed)

    g_val = torch.Generator()
    g_val.manual_seed(seed + 999)

    train_ds = TensorDataset(splits.x_train, splits.y_train)
    val_ds = TensorDataset(splits.x_val, splits.y_val)
    test_ds = TensorDataset(splits.x_test, splits.y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last_train,
        generator=g_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        generator=g_val,
    )

    train_eval_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    test_eval_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    return train_loader, val_loader, train_eval_loader, test_eval_loader


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def eval_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.eval()
    ce = nn.CrossEntropyLoss()
    losses: List[float] = []
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = net(xb)
        loss = ce(logits, yb)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))
    return (float(np.mean(losses)) if losses else float("nan"), correct / max(total, 1))


# -------------------------
# F* helper functions (faithful to your scripts)
# -------------------------
def clamp_eta(eta: torch.Tensor, *, step: int, warmup_steps: int, c_base: float, Lmin: float,
              eta_min: float, eta_max: float, eps: float) -> torch.Tensor:
    if step < warmup_steps:
        warmup_max = min(eta_max, 1.2 * c_base / (Lmin + eps))
        return torch.clamp(eta, min=eta_min, max=warmup_max)
    return torch.clamp(eta, min=eta_min, max=eta_max)


def global_norm(tensors: List[torch.Tensor], eps: float) -> torch.Tensor:
    s = torch.zeros([], device=tensors[0].device, dtype=torch.float32)
    for t in tensors:
        s = s + (t.detach().float() ** 2).sum()
    return torch.sqrt(s + eps)


# -------------------------
# Training loops per method
# -------------------------
def train_sgd_adam_rmsprop(
    method: str,
    net: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    test_eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    wandb_run,
):
    ce = nn.CrossEntropyLoss()
    if method == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif method == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif method == "rmsprop":
        opt = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(method)

    net.to(device)

    for epoch in range(epochs):
        net.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            train_loss_sum += float(loss.item()) * int(yb.size(0))
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        # per-epoch metrics (train from streaming, test from eval)
        train_loss = train_loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        test_loss, test_acc = eval_model(net, test_eval_loader, device)

        # (optional) also compute full-train eval for consistency
        # train_loss_eval, train_acc_eval = eval_model(net, train_eval_loader, device)

        print(
            f"[{method.upper()} EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )


def train_f1(
    net: nn.Module,
    learner: LearnedL_F1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    c_base: float,
    eps: float,
    Lmin: float,
    Lmax: float,
    warmup_steps: int,
    eta_min: float,
    eta_max: float,
    theta_lr: float,
    clip_grad: float,
    wandb_run,
):
    ce = nn.CrossEntropyLoss()
    net.to(device)
    learner.to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=theta_lr)

    params = list(net.parameters())
    val_iter = infinite_loader(val_loader)
    global_step = 0

    for epoch in range(epochs):
        net.train()
        learner.train()

        train_loss_sum = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # --- 1) train batch: g_t ---
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)

            grads = torch.autograd.grad(train_loss, params, create_graph=False, retain_graph=False)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # phi = log ||g||
            g_norm = global_norm(grads, eps=eps)
            phi = torch.log(g_norm + eps).view(1, 1)

            # --- 2) val batch: grad_val ---
            xv, yv = next(val_iter)
            xv = xv.to(device)
            yv = yv.to(device)

            logits_val = net(xv)
            val_loss = ce(logits_val, yv)
            grad_val = torch.autograd.grad(val_loss, params, create_graph=False, retain_graph=False)
            grad_val = [gv if gv is not None else torch.zeros_like(p) for gv, p in zip(grad_val, params)]

            # dot = <grad_val, stop_grad(g)>
            dot = torch.zeros([], device=device, dtype=torch.float32)
            for gv, g in zip(grad_val, grads):
                dot = dot + (gv.float() * g.detach().float()).sum()

            # --- 3) meta update theta ---
            L_theta = learner(phi)
            eta = c_base / (L_theta + eps)
            eta = clamp_eta(eta, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                            eta_min=eta_min, eta_max=eta_max, eps=eps)
            eta_scalar = eta.squeeze()

            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(torch.square(torch.log(L_theta + eps)))

            theta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            theta_opt.step()

            # --- 4) clip g and update w using updated theta ---
            g_norm_for_clip = global_norm(grads, eps=1e-12)
            if clip_grad > 0.0 and float(g_norm_for_clip.item()) > clip_grad:
                clip_coef = clip_grad / float(g_norm_for_clip.item())
            else:
                clip_coef = 1.0

            with torch.no_grad():
                L_now = learner(phi)
                eta_now = c_base / (L_now + eps)
                eta_now = clamp_eta(eta_now, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                                    eta_min=eta_min, eta_max=eta_max, eps=eps)
                eta_now = eta_now.squeeze()

                for p, g in zip(params, grads):
                    p.data -= eta_now.to(p.device).to(p.dtype) * (g * clip_coef)

            # streaming train metrics
            train_loss_sum += float(train_loss.item()) * int(yb.size(0))
            preds = logits_tr.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

            global_step += 1

        train_loss_epoch = train_loss_sum / max(total, 1)
        train_acc_epoch = correct / max(total, 1)
        test_loss, test_acc = eval_model(net, test_eval_loader, device)

        print(
            f"[F1 EPOCH {epoch}] "
            f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )


def train_f2(
    net: nn.Module,
    learner: LearnedL_F2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    c_base: float,
    eps: float,
    Lmin: float,
    warmup_steps: int,
    eta_min: float,
    eta_max: float,
    theta_lr: float,
    clip_grad: float,
    beta: float,
    wandb_run,
):
    ce = nn.CrossEntropyLoss()
    net.to(device)
    learner.to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=theta_lr)

    params = list(net.parameters())
    momentum = [torch.zeros_like(p) for p in params]

    val_iter = infinite_loader(val_loader)
    global_step = 0

    for epoch in range(epochs):
        net.train()
        learner.train()

        train_loss_sum = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # --- 1) train batch: g_t ---
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)

            grads = torch.autograd.grad(train_loss, params, create_graph=False, retain_graph=False)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # --- 2) update momentum ---
            with torch.no_grad():
                for m, g in zip(momentum, grads):
                    m.mul_(beta).add_(g, alpha=(1.0 - beta))

            g_norm = global_norm(grads, eps=eps)
            m_norm = global_norm(momentum, eps=eps)
            phi = torch.stack([torch.log(g_norm + eps), torch.log(m_norm + eps)], dim=0).view(1, 2)

            # --- 3) val batch: grad_val ---
            xv, yv = next(val_iter)
            xv = xv.to(device)
            yv = yv.to(device)

            logits_val = net(xv)
            val_loss = ce(logits_val, yv)
            grad_val = torch.autograd.grad(val_loss, params, create_graph=False, retain_graph=False)
            grad_val = [gv if gv is not None else torch.zeros_like(p) for gv, p in zip(grad_val, params)]

            # dot = <grad_val, stop_grad(m)>
            dot = torch.zeros([], device=device, dtype=torch.float32)
            for gv, m in zip(grad_val, momentum):
                dot = dot + (gv.float() * m.detach().float()).sum()

            # --- 4) meta update theta ---
            L_theta = learner(phi)
            eta = c_base / (L_theta + eps)
            eta = clamp_eta(eta, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                            eta_min=eta_min, eta_max=eta_max, eps=eps)
            eta_scalar = eta.squeeze()

            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(torch.square(torch.log(L_theta + eps)))

            theta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            theta_opt.step()

            # --- 5) clip momentum and update w ---
            m_norm_for_clip = global_norm(momentum, eps=1e-12)
            if clip_grad > 0.0 and float(m_norm_for_clip.item()) > clip_grad:
                clip_coef = clip_grad / float(m_norm_for_clip.item())
            else:
                clip_coef = 1.0

            with torch.no_grad():
                L_now = learner(phi)
                eta_now = c_base / (L_now + eps)
                eta_now = clamp_eta(eta_now, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                                    eta_min=eta_min, eta_max=eta_max, eps=eps)
                eta_now = eta_now.squeeze()

                for p, m in zip(params, momentum):
                    p.data -= eta_now.to(p.device).to(p.dtype) * (m * clip_coef)

            # streaming train metrics
            train_loss_sum += float(train_loss.item()) * int(yb.size(0))
            preds = logits_tr.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

            global_step += 1

        train_loss_epoch = train_loss_sum / max(total, 1)
        train_acc_epoch = correct / max(total, 1)
        test_loss, test_acc = eval_model(net, test_eval_loader, device)

        print(
            f"[F2 EPOCH {epoch}] "
            f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )


def train_f3_or_f3alpha(
    method: str,  # "f3" or "f3alpha"
    net: nn.Module,
    learner: LearnedL_F3,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    c_base: float,
    eps: float,
    Lmin: float,
    warmup_steps: int,
    eta_min: float,
    eta_max: float,
    theta_lr: float,
    clip_grad: float,
    beta: float,
    alpha_mix: float,
    wandb_run,
):
    ce = nn.CrossEntropyLoss()
    net.to(device)
    learner.to(device)
    theta_opt = torch.optim.Adam(learner.parameters(), lr=theta_lr)

    params = list(net.parameters())
    momentum = [torch.zeros_like(p) for p in params]

    val_iter = infinite_loader(val_loader)
    global_step = 0
    alpha_mix = float(np.clip(alpha_mix, 0.0, 1.0))

    for epoch in range(epochs):
        net.train()
        learner.train()

        train_loss_sum = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # --- 1) train batch: g_t ---
            logits_tr = net(xb)
            train_loss = ce(logits_tr, yb)

            grads = torch.autograd.grad(train_loss, params, create_graph=False, retain_graph=False)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # --- 2) update momentum ---
            with torch.no_grad():
                for m, g in zip(momentum, grads):
                    m.mul_(beta).add_(g, alpha=(1.0 - beta))

            # --- 3) build features ---
            if method == "f3":
                g_norm = global_norm(grads, eps=eps)
                m_norm = global_norm(momentum, eps=eps)
                mg_diff = global_norm([m.detach() - g.detach() for m, g in zip(momentum, grads)], eps=eps)

                phi = torch.stack(
                    [torch.log(g_norm + eps), torch.log(m_norm + eps), mg_diff],
                    dim=0
                ).view(1, 3)

            elif method == "f3alpha":
                a = alpha_mix
                g_scaled = [(1.0 - a) * g.detach() for g in grads]
                m_scaled = [a * m.detach() for m in momentum]

                g_norm_s = global_norm(g_scaled, eps=eps)
                m_norm_s = global_norm(m_scaled, eps=eps)
                mg_diff = global_norm([m.detach() - g.detach() for m, g in zip(momentum, grads)], eps=eps)

                phi = torch.stack(
                    [torch.log(g_norm_s + eps), torch.log(m_norm_s + eps), mg_diff],
                    dim=0
                ).view(1, 3)
            else:
                raise ValueError(method)

            # --- 4) val batch ---
            xv, yv = next(val_iter)
            xv = xv.to(device)
            yv = yv.to(device)

            logits_val = net(xv)
            val_loss = ce(logits_val, yv)
            grad_val = torch.autograd.grad(val_loss, params, create_graph=False, retain_graph=False)
            grad_val = [gv if gv is not None else torch.zeros_like(p) for gv, p in zip(grad_val, params)]

            # dot = <grad_val, stop_grad(m)>
            dot = torch.zeros([], device=device, dtype=torch.float32)
            for gv, m in zip(grad_val, momentum):
                dot = dot + (gv.float() * m.detach().float()).sum()

            # --- 5) meta update theta ---
            L_theta = learner(phi)
            eta = c_base / (L_theta + eps)
            eta = clamp_eta(eta, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                            eta_min=eta_min, eta_max=eta_max, eps=eps)
            eta_scalar = eta.squeeze()

            meta_loss = val_loss.detach() - eta_scalar * dot.detach()
            meta_loss = meta_loss + 1e-4 * torch.mean(torch.square(torch.log(L_theta + eps)))

            theta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            theta_opt.step()

            # --- 6) clip momentum and update w ---
            m_norm_for_clip = global_norm(momentum, eps=1e-12)
            if clip_grad > 0.0 and float(m_norm_for_clip.item()) > clip_grad:
                clip_coef = clip_grad / float(m_norm_for_clip.item())
            else:
                clip_coef = 1.0

            with torch.no_grad():
                L_now = learner(phi)
                eta_now = c_base / (L_now + eps)
                eta_now = clamp_eta(eta_now, step=global_step, warmup_steps=warmup_steps, c_base=c_base, Lmin=Lmin,
                                    eta_min=eta_min, eta_max=eta_max, eps=eps)
                eta_now = eta_now.squeeze()

                for p, m in zip(params, momentum):
                    p.data -= eta_now.to(p.device).to(p.dtype) * (m * clip_coef)

            # streaming train metrics
            train_loss_sum += float(train_loss.item()) * int(yb.size(0))
            preds = logits_tr.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

            global_step += 1

        train_loss_epoch = train_loss_sum / max(total, 1)
        train_acc_epoch = correct / max(total, 1)
        test_loss, test_acc = eval_model(net, test_eval_loader, device)

        tag = "F3" if method == "f3" else "F3+alpha"
        print(
            f"[{tag} EPOCH {epoch}] "
            f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )


# -------------------------
# Orchestration / CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)

    p.add_argument(
        "--methods",
        type=str,
        default="sgd,adam,rmsprop,f1,f2,f3,f3alpha",
        help="comma-separated: sgd,adam,rmsprop,f1,f2,f3,f3alpha",
    )

    # Baseline optimizer hparams
    p.add_argument("--lr_sgd", type=float, default=0.01)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_rmsprop", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # F* hparams (aligned with your scripts)
    p.add_argument("--c_base", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--Lmin", type=float, default=1e-3)
    p.add_argument("--Lmax", type=float, default=1e3)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--eta_max", type=float, default=1.0)
    p.add_argument("--theta_lr", type=float, default=1e-3)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--alpha_mix", type=float, default=1.0, help="only used for f3alpha")

    p.add_argument("--learned_hidden", type=int, default=32, help="hidden size for LearnedL MLPs")

    # WandB
    p.add_argument("--wandb", action="store_true", help="enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="l2o-mnist1d")
    p.add_argument("--wandb_group", type=str, default="mnist1d_eval_compare")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument(
        "--wandb_mode",
        type=str,
        default="multi",
        choices=["multi", "single"],
        help="multi: one run per method (recommended). single: one run with method/* metrics.",
    )

    return p.parse_args()


def init_wandb(args: argparse.Namespace, method: Optional[str] = None):
    if not args.wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but --wandb was passed.")

    if args.wandb_mode == "single":
        # only init once in main
        raise RuntimeError("init_wandb(method=...) should not be called in single mode.")

    run_name = args.wandb_name
    if run_name is None:
        run_name = f"mnist1d_{method}_data{args.data_seed}_seed{args.seed}"

    cfg = vars(args).copy()
    cfg["method"] = method

    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=cfg,
        reinit=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    allowed = {"sgd", "adam", "rmsprop", "f1", "f2", "f3", "f3alpha"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown method '{m}'. Allowed: {sorted(allowed)}")

    # load data once (CPU tensors); dataloaders created per-method with fixed seed
    splits = make_splits(preprocess_dir=args.preprocess_dir, data_seed=args.data_seed, as_conv1d=True)

    if args.wandb and args.wandb_mode == "single":
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was passed.")
        run_name = args.wandb_name or f"mnist1d_compare_data{args.data_seed}_seed{args.seed}"
        wandb_run_single = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            config=vars(args),
        )
    else:
        wandb_run_single = None

    for method in methods:
        # reset seeds for fair init + shuffle
        set_seed(args.seed, args.data_seed)

        # loaders (deterministic shuffle with generators)
        train_loader, val_loader, train_eval_loader, test_eval_loader = make_loaders(
            splits=splits,
            bs=args.bs,
            seed=args.seed,
            drop_last_train=True,
        )

        # create model(s)
        num_classes = int(torch.max(splits.y_train).item() + 1)
        net = Conv1DMNIST1D(length=40, num_classes=num_classes)

        # wandb run
        if args.wandb_mode == "multi":
            wb = init_wandb(args, method=method)
        else:
            wb = wandb_run_single

        # prefix for single-run logging
        if wb is not None and args.wandb_mode == "single":
            # wrap wandb.log to add prefix
            orig_log = wandb.log

            def _log_with_prefix(d: Dict):
                dd = {}
                for k, v in d.items():
                    if k in ("epoch",):
                        dd[k] = v
                    else:
                        dd[f"{method}/{k}"] = v
                orig_log(dd)

            wandb.log = _log_with_prefix  # type: ignore

        # dispatch
        t0 = time.time()
        if method in {"sgd", "adam", "rmsprop"}:
            lr = {"sgd": args.lr_sgd, "adam": args.lr_adam, "rmsprop": args.lr_rmsprop}[method]
            train_sgd_adam_rmsprop(
                method=method,
                net=net,
                train_loader=train_loader,
                train_eval_loader=train_eval_loader,
                test_eval_loader=test_eval_loader,
                device=device,
                epochs=args.epochs,
                lr=lr,
                weight_decay=args.weight_decay,
                wandb_run=wb,
            )

        elif method == "f1":
            learner = LearnedL_F1(L_min=args.Lmin, L_max=args.Lmax, hidden=args.learned_hidden)
            train_f1(
                net=net,
                learner=learner,
                train_loader=train_loader,
                val_loader=val_loader,
                test_eval_loader=test_eval_loader,
                device=device,
                epochs=args.epochs,
                c_base=args.c_base,
                eps=args.eps,
                Lmin=args.Lmin,
                Lmax=args.Lmax,
                warmup_steps=args.warmup_steps,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                theta_lr=args.theta_lr,
                clip_grad=args.clip_grad,
                wandb_run=wb,
            )

        elif method == "f2":
            learner = LearnedL_F2(L_min=args.Lmin, L_max=args.Lmax, hidden=args.learned_hidden)
            train_f2(
                net=net,
                learner=learner,
                train_loader=train_loader,
                val_loader=val_loader,
                test_eval_loader=test_eval_loader,
                device=device,
                epochs=args.epochs,
                c_base=args.c_base,
                eps=args.eps,
                Lmin=args.Lmin,
                warmup_steps=args.warmup_steps,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                theta_lr=args.theta_lr,
                clip_grad=args.clip_grad,
                beta=args.beta,
                wandb_run=wb,
            )

        elif method in {"f3", "f3alpha"}:
            learner = LearnedL_F3(L_min=args.Lmin, L_max=args.Lmax, hidden=args.learned_hidden)
            train_f3_or_f3alpha(
                method=method,
                net=net,
                learner=learner,
                train_loader=train_loader,
                val_loader=val_loader,
                test_eval_loader=test_eval_loader,
                device=device,
                epochs=args.epochs,
                c_base=args.c_base,
                eps=args.eps,
                Lmin=args.Lmin,
                warmup_steps=args.warmup_steps,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                theta_lr=args.theta_lr,
                clip_grad=args.clip_grad,
                beta=args.beta,
                alpha_mix=args.alpha_mix,
                wandb_run=wb,
            )

        else:
            raise RuntimeError("unreachable")

        print(f"[DONE] method={method} elapsed={time.time()-t0:.1f}s")

        if wb is not None and args.wandb_mode == "multi":
            wb.finish()

    if wandb_run_single is not None:
        wandb_run_single.finish()


if __name__ == "__main__":
    main()

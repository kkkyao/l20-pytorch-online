#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-style coordinate-wise learned optimizer (2-layer LSTM) on CIFAR-10 + ResNet18 (CIFAR variant).

Key points (close to "Learning to learn by gradient descent"):
  - Optimizee parameters: w (ResNet18 weights)
  - Optimizer parameters: phi (LSTM weights), meta-trained via unroll + truncated BPTT
  - Coordinate-wise decomposition: each parameter coordinate has its own (h, c)
  - CIFAR paper tweak: separate optimizer for conv-params vs fc-params (two LSTMs)
  - First-order approximation: detach gradients before feeding into LSTM (avoid 2nd derivatives)

Engineering:
  - Optional fp16 state for h/c to reduce memory (--state_fp16)
  - Chunked LSTM forward over coordinates (--chunk_size)
  - Uses stateless functional_call to keep w differentiable across unroll, including BN buffers mutation.

This script is intended as a "can it run" faithful prototype. Expect it to be slow for large settings.
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from data_cifar10 import load_cifar10
from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ----------------------------- Utils -----------------------------

def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise AssertionError(f"Expected CIFAR-10 images with 4 dims, got {x.shape}")

    if x.shape[1] == 3:
        x_nchw = x
    elif x.shape[-1] == 3:
        x_nchw = np.transpose(x, (0, 3, 1, 2))
    else:
        raise AssertionError(f"Unexpected CIFAR-10 shape {x.shape}")

    mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x_nchw - mean) / std


def evaluate(model: nn.Module, params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor],
             loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Stateless eval: functional_call with provided params/buffers.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    # functional_call import (PyTorch 2.x)
    try:
        from torch.nn.utils.stateless import functional_call
    except Exception as e:
        raise RuntimeError("This script requires torch.nn.utils.stateless.functional_call (PyTorch 2.x).") from e

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = functional_call(model, (params, buffers), (xb,))
            loss = ce(logits, yb)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

    return (float(np.mean(losses)) if losses else float("nan"),
            correct / max(total, 1))


# ---------------------- ResNet18 (CIFAR) ----------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
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


def build_model() -> nn.Module:
    return ResNet18CIFAR(num_classes=10)


# ---------------------- Gradient preprocessing (Appendix A style) ----------------------

def grad_preprocess(g: torch.Tensor, factor: float = 10.0) -> torch.Tensor:
    """
    g: [N,1] detached gradient
    returns: [N,2] preprocessed
    """
    # threshold = exp(-factor)
    thr = float(np.exp(-factor))
    gd = g  # already detached
    out = torch.zeros(gd.size(0), 2, device=gd.device, dtype=gd.dtype)
    keep = (gd.abs() >= thr).view(-1)

    out[keep, 0] = (torch.log(gd[keep].abs() + 1e-8) / factor).view(-1)
    out[keep, 1] = torch.sign(gd[keep]).view(-1)

    out[~keep, 0] = -1.0
    out[~keep, 1] = (float(np.exp(factor)) * gd[~keep]).view(-1)
    return out


# ---------------------- Coordinate-wise 2-layer LSTM Optimizer ----------------------

class CoordWise2LayerLSTM(nn.Module):
    def __init__(self, hidden_sz: int = 20, preproc: bool = True, preproc_factor: float = 10.0):
        super().__init__()
        self.hidden_sz = int(hidden_sz)
        self.preproc = bool(preproc)
        self.preproc_factor = float(preproc_factor)

        in_dim = 2 if self.preproc else 1
        self.lstm1 = nn.LSTMCell(in_dim, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz, 1)

    def forward_chunked(self,
                        g_vec: torch.Tensor,
                        h: Tuple[torch.Tensor, torch.Tensor],
                        c: Tuple[torch.Tensor, torch.Tensor],
                        chunk_size: int = 500_000) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        g_vec: [N,1] (detached)
        h: (h0,h1), each [N,H]
        c: (c0,c1), each [N,H]
        returns:
          u: [N,1]
          h_new, c_new
        """
        N = g_vec.size(0)
        H = self.hidden_sz

        u_out = torch.empty(N, 1, device=g_vec.device, dtype=g_vec.dtype)
        h0_new = torch.empty(N, H, device=g_vec.device, dtype=h[0].dtype)
        h1_new = torch.empty(N, H, device=g_vec.device, dtype=h[1].dtype)
        c0_new = torch.empty(N, H, device=g_vec.device, dtype=c[0].dtype)
        c1_new = torch.empty(N, H, device=g_vec.device, dtype=c[1].dtype)

        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)

            g_chunk = g_vec[s:e]  # [B,1]
            if self.preproc:
                x = grad_preprocess(g_chunk, factor=self.preproc_factor)  # [B,2]
            else:
                x = g_chunk

            h0, c0 = self.lstm1(x, (h[0][s:e].to(x.dtype), c[0][s:e].to(x.dtype)))
            h1, c1 = self.lstm2(h0, (h[1][s:e].to(h0.dtype), c[1][s:e].to(h0.dtype)))
            u = self.out(h1)  # [B,1]

            u_out[s:e] = u
            h0_new[s:e] = h0.to(h0_new.dtype)
            h1_new[s:e] = h1.to(h1_new.dtype)
            c0_new[s:e] = c0.to(c0_new.dtype)
            c1_new[s:e] = c1.to(c1_new.dtype)

        return u_out, (h0_new, h1_new), (c0_new, c1_new)


@dataclass
class GroupMeta:
    names: List[str]
    sizes: List[int]
    shapes: List[torch.Size]
    total: int


def build_group_meta(params: Dict[str, torch.Tensor], names: List[str]) -> GroupMeta:
    sizes, shapes = [], []
    total = 0
    for n in names:
        p = params[n]
        sizes.append(p.numel())
        shapes.append(p.shape)
        total += p.numel()
    return GroupMeta(names=names, sizes=sizes, shapes=shapes, total=total)


def flatten_from_dict(grads: Dict[str, torch.Tensor], meta: GroupMeta) -> torch.Tensor:
    parts = []
    for n, sz in zip(meta.names, meta.sizes):
        g = grads[n]
        parts.append(g.reshape(-1))
    return torch.cat(parts, dim=0).view(-1, 1)


def apply_updates(params: Dict[str, torch.Tensor], meta: GroupMeta, upd_vec: torch.Tensor, out_mul: float) -> Dict[str, torch.Tensor]:
    """
    params[name] <- params[name] + out_mul * upd_slice (differentiable)
    returns new params dict (updated tensors)
    """
    new_params = dict(params)
    v = upd_vec.view(-1)
    offset = 0
    for n, sz, shp in zip(meta.names, meta.sizes, meta.shapes):
        sl = v[offset:offset + sz].view(shp)
        new_params[n] = params[n] + (float(out_mul) * sl)
        offset += sz
    return new_params


def detach_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in params.items():
        out[k] = v.detach().requires_grad_(True)
    return out


def clone_buffers(buffers: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # buffers are not differentiable; we deep copy tensors to isolate episodes
    out = {}
    for k, v in buffers.items():
        out[k] = v.detach().clone()
    return out


# ----------------------------- Main meta-training loop -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # data / run
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5, help="meta-epochs")
    ap.add_argument("--episodes_per_epoch", type=int, default=1, help="episodes (tasks) per meta-epoch")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/cifar10_conv2d_preprocess")

    # inner optimization trajectory
    ap.add_argument("--inner_steps", type=int, default=20, help="total inner steps per episode")
    ap.add_argument("--unroll", type=int, default=5, help="truncated BPTT window length")
    ap.add_argument("--out_mul", type=float, default=1e-3, help="scale multiplier for LSTM output updates")

    # learned optimizer (paper-style)
    ap.add_argument("--hidden_sz", type=int, default=5, help="LSTM hidden size (paper used 20; start small)")
    ap.add_argument("--preproc", action="store_true", help="enable Appendix-A grad preprocessing")
    ap.add_argument("--preproc_factor", type=float, default=10.0)

    # engineering
    ap.add_argument("--chunk_size", type=int, default=500_000, help="chunk size for coord-wise LSTM forward")
    ap.add_argument("--state_fp16", action="store_true", help="store h/c in fp16 to save memory")
    ap.add_argument("--phi_lr", type=float, default=1e-3, help="outer Adam lr for optimizer parameters")
    ap.add_argument("--clip_phi_grad", type=float, default=1.0, help="grad clip for optimizer params")
    ap.add_argument("--device", type=str, default="cuda")

    # logging
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-cifar10")
    ap.add_argument("--wandb_group", type=str, default="cifar10_resnet18_paperstyle_lstm")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"[INFO] device={device}")

    run_name = (
        f"cifar10_resnet18_paperstyle_lstm_data{args.data_seed}_seed{args.seed}_"
        f"hs{args.hidden_sz}_un{args.unroll}_T{args.inner_steps}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir={run_dir}")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb is passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    # functional_call import
    try:
        from torch.nn.utils.stateless import functional_call
    except Exception as e:
        raise RuntimeError("This script requires torch.nn.utils.stateless.functional_call (PyTorch 2.x).") from e

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_cifar10()
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

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    cfg = LoaderCfg(batch_size=args.bs, num_workers=0)
    train_loader, val_loader = make_train_val_loaders(
        train_ds, val_ds, cfg,
        seed=args.seed, train_shuffle=True, val_shuffle=True,
        train_drop_last=True, val_drop_last=True
    )
    # inner-loop uses train only; val/test for reporting
    val_eval_loader = make_eval_loader(val_ds, batch_size=256, num_workers=0, pin_memory=False)
    test_eval_loader = make_eval_loader(test_ds, batch_size=256, num_workers=0, pin_memory=False)

    def infinite_loader(loader):
        while True:
            for b in loader:
                yield b

    train_iter = infinite_loader(train_loader)

    # ---------------- models ----------------
    base_model = build_model().to(device)
    base_model.train()

    # Build initial params/buffers templates (names)
    init_params = {k: v.detach().clone().requires_grad_(True) for k, v in base_model.named_parameters()}
    init_buffers = {k: v.detach().clone() for k, v in base_model.named_buffers()}

    # Group split: conv-group = all except final fc; fc-group = fc.*
    fc_names = [n for n in init_params.keys() if n.startswith("fc.")]
    conv_names = [n for n in init_params.keys() if n not in fc_names]

    conv_meta = build_group_meta(init_params, conv_names)
    fc_meta = build_group_meta(init_params, fc_names)

    print(f"[INFO] #params total: {sum(p.numel() for p in init_params.values())}")
    print(f"[INFO] conv-group coords: {conv_meta.total}, fc-group coords: {fc_meta.total}")

    # Two optimizers like paper CIFAR tweak
    opt_conv = CoordWise2LayerLSTM(hidden_sz=args.hidden_sz, preproc=args.preproc, preproc_factor=args.preproc_factor).to(device)
    opt_fc = CoordWise2LayerLSTM(hidden_sz=args.hidden_sz, preproc=args.preproc, preproc_factor=args.preproc_factor).to(device)

    outer_opt = torch.optim.Adam(list(opt_conv.parameters()) + list(opt_fc.parameters()), lr=args.phi_lr)
    ce = nn.CrossEntropyLoss()

    # Logs
    log_path = run_dir / "meta_log.csv"
    with open(log_path, "w") as f:
        f.write("meta_epoch,episode,inner_step,unroll_step,loss\n")

    start_time = time.time()

    # ---------------- meta-training ----------------
    for me in range(args.epochs):
        base_model.train()
        meta_losses = []

        for ep in range(args.episodes_per_epoch):
            # fresh episode: reset optimizee params and BN buffers
            params = {k: v.detach().clone().requires_grad_(True) for k, v in init_params.items()}
            buffers = clone_buffers(init_buffers)

            # init states
            st_dtype = torch.float16 if args.state_fp16 else torch.float32
            H = args.hidden_sz

            h_conv = (torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype),
                      torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype))
            c_conv = (torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype),
                      torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype))

            h_fc = (torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype),
                    torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype))
            c_fc = (torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype),
                    torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype))

            inner_steps = int(args.inner_steps)
            unroll = int(args.unroll)
            assert inner_steps >= 1 and unroll >= 1
            n_windows = (inner_steps + unroll - 1) // unroll

            for widx in range(n_windows):
                # collect losses in this window
                window_losses = []
                steps_in_window = min(unroll, inner_steps - widx * unroll)

                for uidx in range(steps_in_window):
                    xb, yb = next(train_iter)
                    xb = xb.to(device)
                    yb = yb.to(device)

                    # forward with functional_call (BN buffers mutate in-place)
                    logits = functional_call(base_model, (params, buffers), (xb,))
                    loss = ce(logits, yb)
                    window_losses.append(loss)

                    # grads wrt params (no 2nd derivatives)
                    grads_list = torch.autograd.grad(loss, list(params.values()), create_graph=False, retain_graph=False)
                    grads = {k: g for (k, _), g in zip(params.items(), grads_list)}
                    # detach grads before feeding into optimizer (paper-style first-order approx)
                    grads = {k: (v.detach()) for k, v in grads.items()}

                    # flatten per group
                    g_conv = flatten_from_dict(grads, conv_meta)  # [Nconv,1]
                    g_fc = flatten_from_dict(grads, fc_meta)      # [Nfc,1]

                    # run coord-wise LSTM (chunked)
                    u_conv, h_conv_new, c_conv_new = opt_conv.forward_chunked(
                        g_conv, h_conv, c_conv, chunk_size=int(args.chunk_size)
                    )
                    u_fc, h_fc_new, c_fc_new = opt_fc.forward_chunked(
                        g_fc, h_fc, c_fc, chunk_size=int(args.chunk_size)
                    )

                    # update params (differentiable wrt optimizer params)
                    params = apply_updates(params, conv_meta, u_conv, out_mul=float(args.out_mul))
                    params = apply_updates(params, fc_meta, u_fc, out_mul=float(args.out_mul))

                    # advance states (keep graph for optimizer params; states themselves do not require grad)
                    h_conv, c_conv = h_conv_new, c_conv_new
                    h_fc, c_fc = h_fc_new, c_fc_new

                    # logging
                    inner_step = widx * unroll + uidx
                    with open(log_path, "a") as f:
                        f.write(f"{me},{ep},{inner_step},{uidx},{float(loss.item()):.8f}\n")

                # meta-update on optimizer params phi using truncated BPTT window loss
                meta_loss = torch.stack(window_losses).mean()
                outer_opt.zero_grad(set_to_none=True)
                meta_loss.backward()

                if args.clip_phi_grad and args.clip_phi_grad > 0:
                    torch.nn.utils.clip_grad_norm_(list(opt_conv.parameters()) + list(opt_fc.parameters()),
                                                   max_norm=float(args.clip_phi_grad))
                outer_opt.step()
                meta_losses.append(float(meta_loss.item()))

                # truncate: detach optimizee params and states
                params = detach_params(params)
                # states are buffers; detach to avoid accidental graph references
                h_conv = (h_conv[0].detach(), h_conv[1].detach())
                c_conv = (c_conv[0].detach(), c_conv[1].detach())
                h_fc = (h_fc[0].detach(), h_fc[1].detach())
                c_fc = (c_fc[0].detach(), c_fc[1].detach())

        # simple reporting (evaluate using current optimizer on one fresh episode for a few steps)
        elapsed = time.time() - start_time
        meta_mean = float(np.mean(meta_losses)) if meta_losses else float("nan")

        # quick eval: run a short inner training with learned optimizer, then test/val
        params_eval = {k: v.detach().clone().requires_grad_(True) for k, v in init_params.items()}
        buffers_eval = clone_buffers(init_buffers)

        # re-init states for eval
        st_dtype = torch.float16 if args.state_fp16 else torch.float32
        H = args.hidden_sz
        h_conv = (torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype),
                  torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype))
        c_conv = (torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype),
                  torch.zeros(conv_meta.total, H, device=device, dtype=st_dtype))
        h_fc = (torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype),
                torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype))
        c_fc = (torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype),
                torch.zeros(fc_meta.total, H, device=device, dtype=st_dtype))

        # take a few steps
        eval_steps = min(20, int(args.inner_steps))
        for _ in range(eval_steps):
            xb, yb = next(train_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = functional_call(base_model, (params_eval, buffers_eval), (xb,))
            loss = ce(logits, yb)
            grads_list = torch.autograd.grad(loss, list(params_eval.values()), create_graph=False, retain_graph=False)
            grads = {k: g.detach() for (k, _), g in zip(params_eval.items(), grads_list)}

            g_conv = flatten_from_dict(grads, conv_meta)
            g_fc = flatten_from_dict(grads, fc_meta)

            with torch.no_grad():
                u_conv, h_conv_new, c_conv_new = opt_conv.forward_chunked(
                    g_conv, h_conv, c_conv, chunk_size=int(args.chunk_size)
                )
                u_fc, h_fc_new, c_fc_new = opt_fc.forward_chunked(
                    g_fc, h_fc, c_fc, chunk_size=int(args.chunk_size)
                )
                params_eval = apply_updates(params_eval, conv_meta, u_conv, out_mul=float(args.out_mul))
                params_eval = apply_updates(params_eval, fc_meta, u_fc, out_mul=float(args.out_mul))
                params_eval = {k: v.detach().requires_grad_(True) for k, v in params_eval.items()}
                h_conv, c_conv = (h_conv_new[0].detach(), h_conv_new[1].detach()), (c_conv_new[0].detach(), c_conv_new[1].detach())
                h_fc, c_fc = (h_fc_new[0].detach(), h_fc_new[1].detach()), (c_fc_new[0].detach(), c_fc_new[1].detach())

        val_loss, val_acc = evaluate(base_model, {k: v.detach() for k, v in params_eval.items()}, buffers_eval, val_eval_loader, device)
        test_loss, test_acc = evaluate(base_model, {k: v.detach() for k, v in params_eval.items()}, buffers_eval, test_eval_loader, device)

        print(f"[META EPOCH {me}] meta_loss_mean={meta_mean:.6f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
              f"elapsed_min={elapsed/60:.2f}")

        if wandb_run is not None:
            wandb.log({
                "meta_epoch": me,
                "meta_loss_mean": meta_mean,
                "eval/val_loss": val_loss,
                "eval/val_acc": val_acc,
                "eval/test_loss": test_loss,
                "eval/test_acc": test_acc,
                "time/elapsed_sec": elapsed,
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

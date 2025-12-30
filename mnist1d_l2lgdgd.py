#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Online learned optimizer baseline (coordinate-wise 2-layer LSTM),
aligned with your F1 pipeline (single-task online meta-learning).

Fixes vs your current version:
  1) META OBJECTIVE SIGN FIX:
       minimize dot = <grad_val(stop), Δw(phi)>
     not -dot. Your previous sign maximized val loss, hence divergence.
  2) Add gradient clipping for optimizer parameters (phi) for stability.
  3) Keep LSTM preprocessing consistent with the paper Appendix A.
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

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ------------------------------- Utilities -------------------------------

class BatchLossLogger:
    def __init__(self, run_dir: Path, meta: dict, filename: str, flush_every: int = 200):
        self.run_dir = Path(run_dir)
        self.meta = meta
        self.flush_every = flush_every
        self.global_iter = 0
        self.curr_epoch = 0
        self.rows = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.curve_path = self.run_dir / filename
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
    pdir = Path(preprocess_dir) / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Preprocess artifacts not found under: {pdir}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mean[None, :]) / (std[None, :] + eps)


def to_NL(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}")


# --------------------------- MLP backbone ---------------------------

def build_mlp(input_len: int = 40, num_layers: int = 5, width: int = 128, activation: str = "relu") -> nn.Module:
    act_layer = nn.ReLU

    class MLPBackbone(nn.Module):
        def __init__(self, length: int):
            super().__init__()
            layers = []
            in_dim = length
            for _ in range(max(0, num_layers)):
                layers.append(nn.Linear(in_dim, width))
                layers.append(act_layer(inplace=True))
                in_dim = width
            self.mlp = nn.Sequential(*layers)
            self.out = nn.Linear(in_dim, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.mlp(x)
            return self.out(x)

    return MLPBackbone(input_len)


# ---------------------- Learned Optimizer ----------------------

class LearnedOptimizer(nn.Module):
    def __init__(self, hidden_sz: int = 20, preproc: bool = True, preproc_factor: float = 10.0):
        super().__init__()
        self.hidden_sz = int(hidden_sz)
        self.preproc = bool(preproc)
        self.preproc_factor = float(preproc_factor)
        self.preproc_threshold = float(np.exp(-self.preproc_factor))

        in_dim = 2 if self.preproc else 1
        self.lstm1 = nn.LSTMCell(in_dim, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz, 1)

    def _preprocess(self, g: torch.Tensor) -> torch.Tensor:
        gd = g.detach()
        out = torch.zeros(gd.size(0), 2, device=gd.device, dtype=gd.dtype)
        keep = (gd.abs() >= self.preproc_threshold).view(-1)

        out[keep, 0] = (torch.log(gd[keep].abs() + 1e-8) / self.preproc_factor).view(-1)
        out[keep, 1] = torch.sign(gd[keep]).view(-1)

        out[~keep, 0] = -1.0
        out[~keep, 1] = (float(np.exp(self.preproc_factor)) * gd[~keep]).view(-1)
        return out

    def forward(self, grad_vec: torch.Tensor, hidden, cell):
        if self.preproc:
            x = self._preprocess(grad_vec)
        else:
            x = grad_vec.detach()
        h0, c0 = self.lstm1(x, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)
        return update, (h0, h1), (c0, c1)


def flatten_like_params(tensors, params):
    vecs = []
    for t, p in zip(tensors, params):
        vecs.append(t.reshape(-1))
    return torch.cat(vecs, dim=0).view(-1, 1)


def unflatten_to_params(vec: torch.Tensor, params):
    outs = []
    offset = 0
    v = vec.view(-1)
    for p in params:
        sz = p.numel()
        outs.append(v[offset:offset + sz].view_as(p))
        offset += sz
    return outs


@torch.no_grad()
def eval_model(net: nn.Module, data_loader, device, ce):
    net.eval()
    losses = []
    correct = 0
    total = 0
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = net(xb)
        loss = ce(logits, yb)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return (float(np.mean(losses)) if losses else float("nan"), correct / max(total, 1))


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")

    ap.add_argument("--mlp_width", type=int, default=128)
    ap.add_argument("--mlp_layers", type=int, default=5)
    ap.add_argument("--activation", type=str, default="relu")

    ap.add_argument("--opt_hidden_sz", type=int, default=20)
    ap.add_argument("--opt_lr", type=float, default=1e-3)
    ap.add_argument("--out_mul", type=float, default=1e-3)
    ap.add_argument("--no_preproc", action="store_true")
    ap.add_argument("--preproc_factor", type=float, default=10.0)

    ap.add_argument("--clip_update", type=float, default=0.1)
    ap.add_argument("--reg_update", type=float, default=1e-6)
    ap.add_argument("--clip_phi_grad", type=float, default=1.0)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-mnist1d")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_mlp_l2l_online")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    set_seed(args.seed)

    run_name = (
        f"mnist1d_mlp_l2l_online_data{args.data_seed}_seed{args.seed}_"
        f"L{args.mlp_layers}_W{args.mlp_width}_"
        f"hs{args.opt_hidden_sz}_lr{args.opt_lr}_out{args.out_mul}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --wandb was passed.")
        wandb_run = wandb.init(
            project="l2o-online(new1)",
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config={
                "dataset": "MNIST-1D",
                "backbone": "MLP",
                "method": "l2l_lstm_online",
                "seed": args.seed,
                "data_seed": args.data_seed,
                "epochs": args.epochs,
                "batch_size": args.bs,
            },
        )

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=args.length, seed=args.data_seed)

    xtr = to_NL(xtr, length=args.length).astype(np.float32, copy=False)
    xte = to_NL(xte, length=args.length).astype(np.float32, copy=False)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)
    std = np.array(norm["std"], np.float32)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train = apply_norm_np(xtr[train_idx], mean, std)
    y_train = ytr[train_idx]
    x_val = apply_norm_np(xtr[val_idx], mean, std)
    y_val = ytr[val_idx]
    x_test = apply_norm_np(xte, mean, std)
    y_test = yte

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )

    def infinite_loader(loader):
        while True:
            for b in loader:
                yield b

    val_iter = infinite_loader(val_loader)

    val_eval_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=512, shuffle=False)
    test_eval_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=512, shuffle=False)

    # ---------------- models ----------------
    net = build_mlp(
        input_len=args.length,
        num_layers=args.mlp_layers,
        width=args.mlp_width,
        activation=args.activation,
    ).to(device)

    opt_net = LearnedOptimizer(
        hidden_sz=args.opt_hidden_sz,
        preproc=(not args.no_preproc),
        preproc_factor=args.preproc_factor,
    ).to(device)

    meta_opt = torch.optim.Adam(opt_net.parameters(), lr=args.opt_lr)
    ce = nn.CrossEntropyLoss()

    params = list(net.parameters())
    n_params = int(sum(p.numel() for p in params))
    hidden = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]

    global_step = 0
    start_time = time.time()

    # ---------------- training loop ----------------
    for epoch in range(args.epochs):
        epoch_start = time.time()

        net.train()
        opt_net.train()

        train_loss_sum = 0.0
        train_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = net(xb)
            train_loss = ce(logits, yb)
            train_loss_sum += float(train_loss.item())
            train_batches += 1

            grads = torch.autograd.grad(train_loss, params, create_graph=False)
            g_vec = flatten_like_params(grads, params)

            update_vec_1, _, _ = opt_net(g_vec, hidden, cell)
            delta_list_1 = unflatten_to_params(update_vec_1 * args.out_mul, params)

            xv, yv = next(val_iter)
            xv = xv.to(device)
            yv = yv.to(device)
            val_logits = net(xv)
            val_loss = ce(val_logits, yv)
            grad_val = torch.autograd.grad(val_loss, params, create_graph=False)

            dot = torch.zeros([], device=device)
            for gv, dw in zip(grad_val, delta_list_1):
                dot = dot + (gv.detach() * dw).sum()

            meta_loss = dot
            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_net.parameters(), args.clip_phi_grad)
            meta_opt.step()

            with torch.no_grad():
                update_vec_2, h_new_2, c_new_2 = opt_net(g_vec, hidden, cell)
                delta_vec = update_vec_2 * args.out_mul
                delta_list = unflatten_to_params(delta_vec, params)
                for p, dw in zip(params, delta_list):
                    p.add_(dw)
                hidden = [h.detach() for h in h_new_2]
                cell = [c.detach() for c in c_new_2]

            global_step += 1

        train_loss_epoch = train_loss_sum / max(train_batches, 1)
        val_loss_epoch, val_acc = eval_model(net, val_eval_loader, device, ce)
        test_loss_epoch, test_acc = eval_model(net, test_eval_loader, device, ce)
        _, train_acc_epoch = eval_model(net, val_eval_loader, device, ce)

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "test_loss": test_loss_epoch,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

        print(
            f"[MNIST1D-L2L EPOCH {epoch}] "
            f"train={train_loss_epoch:.4f} "
            f"test={test_loss_epoch:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

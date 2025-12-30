#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10: Online learned optimizer baseline (coordinate-wise 2-layer LSTM),
single-task online meta-learning (F1-style pipeline analog).

Meta objective (first-order / stop-grad through val grad):
  minimize dot = <grad_val(stop), Δθ(phi)>
so that L_val(θ + Δθ) ≈ L_val(θ) + dot decreases.

Key engineering choices:
  - Use the SAME delta (including clipping) for meta-loss and for the actual apply.
  - Differentiable global-norm clipping for delta in meta objective.
  - Clip gradients of optimizer parameters (phi) for stability.
  - Coordinate-wise 2-layer LSTM + gradient preprocessing (Appendix A in paper).

Notes:
  - This is NOT the paper's main setting (paper mainly meta-trains across a task distribution).
    It is the "online single-task" variant you asked for.
  - Choose a SMALL optimizee (TinyCNN) to make coordinate-wise states feasible.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def clip_delta_vec(delta_vec: torch.Tensor, clip_update: Optional[float], eps: float = 1e-12):
    norm = torch.linalg.vector_norm(delta_vec.view(-1)).clamp_min(eps)
    if clip_update is None or float(clip_update) <= 0.0:
        coef = torch.ones((), device=delta_vec.device, dtype=delta_vec.dtype)
        return delta_vec, norm, coef
    clip_t = torch.tensor(float(clip_update), device=delta_vec.device, dtype=delta_vec.dtype)
    coef = torch.clamp(clip_t / norm, max=1.0)
    return delta_vec * coef, norm, coef


def flatten_like_params(tensors, params):
    vecs = []
    for t, p in zip(tensors, params):
        if t is None:
            vecs.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
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
def eval_model(net: nn.Module, data_loader: DataLoader, device: torch.device, ce: nn.Module):
    net.eval()
    losses = []
    correct = 0
    total = 0
    for xb, yb in data_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = net(xb)
        loss = ce(logits, yb)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return (float(np.mean(losses)) if losses else float("nan"), correct / max(total, 1))


def infinite_loader(loader: DataLoader):
    while True:
        for b in loader:
            yield b


# --------------------------- Optimizee (Tiny CNN) ---------------------------

class TinyCNN(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 32, c3: int = 64, fc: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


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
        x = self._preprocess(grad_vec) if self.preproc else grad_vec.detach()
        h0, c0 = self.lstm1(x, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)
        return update, (h0, h1), (c0, c1)


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--c3", type=int, default=64)
    ap.add_argument("--fc", type=int, default=128)

    ap.add_argument("--opt_hidden_sz", type=int, default=20)
    ap.add_argument("--opt_lr", type=float, default=1e-3)
    ap.add_argument("--out_mul", type=float, default=1e-3)
    ap.add_argument("--no_preproc", action="store_true")
    ap.add_argument("--preproc_factor", type=float, default=10.0)

    ap.add_argument("--clip_update", type=float, default=0.1)
    ap.add_argument("--reg_update", type=float, default=1e-6)
    ap.add_argument("--clip_phi_grad", type=float, default=1.0)

    ap.add_argument("--reset_state_each_epoch", action="store_true")
    ap.add_argument("--state_reset_interval", type=int, default=0)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-online(new1)")
    ap.add_argument("--wandb_group", type=str, default="cifar10_l2lgdgd_online")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    run_name = (
        f"cifar10_tinycnn_l2lgdgd_online_seed{args.seed}_"
        f"c{args.c1}-{args.c2}-{args.c3}_fc{args.fc}_"
        f"hs{args.opt_hidden_sz}_lr{args.opt_lr}_out{args.out_mul}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ======================= WandB (ONLY MODIFIED PART) =======================
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity="leyao-li-epfl",
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config={
                "dataset": "CIFAR-10",
                "backbone": "TinyCNN",
                "method": "learned",
                "seed": int(args.seed),
                "optimizer": "coordinatewise_lstm",
                "regime": "online",
                "epochs": int(args.epochs),
                "batch_size": int(args.bs),
            },
        )
    # ========================================================================

    # ---------------- data ----------------
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tf)

    n_total = len(full_train)
    n_val = int(round(float(args.val_ratio) * n_total))
    n_train = n_total - n_val
    gsplit = torch.Generator().manual_seed(int(args.seed))
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=gsplit)

    val_set = Subset(
        datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=test_tf),
        indices=val_set.indices,
    )

    gen = torch.Generator().manual_seed(int(args.seed))

    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, generator=gen
    )
    val_loader = DataLoader(
        val_set, batch_size=args.bs, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, generator=gen
    )
    val_iter = infinite_loader(val_loader)

    val_eval_loader = DataLoader(val_set, batch_size=512, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    test_eval_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    net = TinyCNN(c1=args.c1, c2=args.c2, c3=args.c3, fc=args.fc).to(device)
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

    curve_logger = BatchLossLogger(
        run_dir,
        meta={"method": "l2lgdgd_online_cifar10", "seed": args.seed, "opt": "lstm_opt", "lr": args.opt_lr},
        filename="curve_l2lgdgd.csv",
    )
    mech_path = run_dir / "mechanism_l2lgdgd.csv"
    train_log_path = run_dir / "train_log_l2lgdgd.csv"
    result_path = run_dir / "result_l2lgdgd.json"

    with open(mech_path, "w", encoding="utf-8") as f:
        f.write("iter,epoch,train_loss,val_dot,meta_loss,upd_norm,upd_norm_preclip,clip_coef,grad_norm\n")
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,elapsed_sec,train_loss,val_loss,test_loss,val_acc,test_acc\n")

    global_step = 0
    start_time = time.time()

    # ---------------- training loop ----------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        curve_logger.on_epoch_begin(epoch)

        net.train()
        opt_net.train()

        if args.reset_state_each_epoch:
            hidden = [torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])]
            cell = [torch.zeros_like(cell[0]), torch.zeros_like(cell[1])]

        train_loss_sum = 0.0
        train_batches = 0

        for xb, yb in train_loader:
            if args.state_reset_interval and args.state_reset_interval > 0:
                if global_step % int(args.state_reset_interval) == 0 and global_step > 0:
                    hidden = [torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])]
                    cell = [torch.zeros_like(cell[0]), torch.zeros_like(cell[1])]

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = net(xb)
            train_loss = ce(logits, yb)
            train_loss_sum += float(train_loss.item())
            train_batches += 1

            grads = torch.autograd.grad(train_loss, params, create_graph=False)
            g_vec = flatten_like_params(grads, params)

            update_vec, h_new, c_new = opt_net(g_vec, hidden, cell)
            delta_vec = update_vec * float(args.out_mul)

            delta_vec_clipped, upd_norm_pre, coef = clip_delta_vec(delta_vec, float(args.clip_update))
            delta_list = unflatten_to_params(delta_vec_clipped, params)

            xv, yv = next(val_iter)
            xv = xv.to(device, non_blocking=True)
            yv = yv.to(device, non_blocking=True)

            val_logits = net(xv)
            val_loss = ce(val_logits, yv)
            grad_val = torch.autograd.grad(val_loss, params, create_graph=False)

            dot = torch.zeros([], device=device, dtype=delta_vec_clipped.dtype)
            for gv, dw in zip(grad_val, delta_list):
                dot = dot + (gv.detach() * dw).sum()

            meta_loss = dot
            if args.reg_update is not None and float(args.reg_update) > 0.0:
                meta_loss = meta_loss + float(args.reg_update) * torch.mean(update_vec.pow(2))

            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()

            if args.clip_phi_grad is not None and float(args.clip_phi_grad) > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(args.clip_phi_grad))

            meta_opt.step()

            with torch.no_grad():
                for p, dw in zip(params, delta_list):
                    p.add_(dw)
                hidden = [h.detach() for h in h_new]
                cell = [c.detach() for c in c_new]

                upd_norm = torch.linalg.vector_norm(delta_vec_clipped.view(-1)).clamp_min(1e-12)
                g_norm = torch.linalg.vector_norm(g_vec.view(-1)).clamp_min(1e-12)

                with open(mech_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{global_step},{epoch},"
                        f"{float(train_loss.item()):.8f},"
                        f"{float(dot.item()):.8f},"
                        f"{float(meta_loss.item()):.8f},"
                        f"{float(upd_norm.item()):.6g},"
                        f"{float(upd_norm_pre.item()):.6g},"
                        f"{float(coef.detach().item()):.6g},"
                        f"{float(g_norm.item()):.6g}\n"
                    )

            curve_logger.on_train_batch_end(float(train_loss.item()))
            global_step += 1

        train_loss_epoch = train_loss_sum / max(train_batches, 1)
        val_loss_epoch, val_acc = eval_model(net, val_eval_loader, device, ce)
        test_loss_epoch, test_acc = eval_model(net, test_eval_loader, device, ce)

        total_elapsed = time.time() - start_time
        epoch_elapsed = time.time() - epoch_start

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{total_elapsed:.3f},"
                f"{train_loss_epoch:.8f},"
                f"{val_loss_epoch:.8f},"
                f"{test_loss_epoch:.8f},"
                f"{val_acc:.6f},"
                f"{test_acc:.6f}\n"
            )

        print(
            f"[CIFAR10-TinyCNN-L2LGDGD-ONLINE EPOCH {epoch}] "
            f"time={epoch_elapsed:.2f}s total={total_elapsed/60:.2f}min "
            f"train={train_loss_epoch:.4f} val={val_loss_epoch:.4f} test={test_loss_epoch:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

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

    final_test_loss, final_test_acc = eval_model(net, test_eval_loader, device, ce)
    print(
        f"[RESULT-CIFAR10-TinyCNN-L2LGDGD-ONLINE] TestAcc={final_test_acc:.4f} "
        f"TestLoss={final_test_loss:.4f} (Total time={total_time/60:.2f} min)"
    )

    result = {
        "dataset": "CIFAR-10",
        "optimizee": "TinyCNN",
        "method": "l2lgdgd_coordinatewise_lstm_online_firstorder",
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "seed": int(args.seed),
        "opt_hidden_sz": int(args.opt_hidden_sz),
        "opt_lr": float(args.opt_lr),
        "out_mul": float(args.out_mul),
        "clip_update": float(args.clip_update),
        "test_acc": float(final_test_acc),
        "test_loss": float(final_test_loss),
        "elapsed_sec": float(total_time),
        "run_dir": str(run_dir),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["final_test_acc"] = float(final_test_acc)
        wandb_run.summary["final_test_loss"] = float(final_test_loss)
        wandb_run.summary["total_time_sec"] = float(total_time)
        for p in [curve_logger.curve_path, mech_path, train_log_path, result_path]:
            if Path(p).exists():
                wandb.save(str(p))
        wandb_run.finish()


if __name__ == "__main__":
    main()

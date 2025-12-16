#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fair baseline for MNIST-1D: same optimizee as L2L-GDGD MLP optimizee.

Optimizee (paper-style):
  40 -> hidden_dim(sigmoid) -> 10
  init_scale default 1e-3 for weights

Data alignment:
  - uses {preprocess_dir}/seed_{data_seed}/split.json and norm.json
  - train = split["train_idx"], val = split["val_idx"], test = xte
  - per-position normalization: (x - mean) / (std + eps)

Training alignment:
  - supports step-based training via --optim_steps (recommended for fair compare to L2L)
  - or epoch-based training via --epochs (if --optim_steps is not provided)

Outputs in runs/<run_name>/:
  - train_log.csv            (periodic evaluation logs)
  - curve.csv                (train loss curve: step -> loss)
  - best.pt                  (best-val checkpoint)
  - result.json              (final summary)
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# ---------------------- utils ----------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_N40(x: np.ndarray, length: int = 40) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}")


def load_split_and_norm(preprocess_dir: Path, data_seed: int) -> Tuple[dict, np.ndarray, np.ndarray]:
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


# ---------------------- model ----------------------
class MLPOptimizeeBaseline(nn.Module):
    """
    40 -> hidden_dim(sigmoid) -> 10
    Weight init: N(0,1)*init_scale (paper-style small init for sigmoid stability)
    """

    def __init__(self, hidden_dim: int = 20, init_scale: float = 1e-3):
        super().__init__()
        self.fc1 = nn.Linear(40, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 10, bias=True)
        self.act = nn.Sigmoid()

        with torch.no_grad():
            self.fc1.weight.normal_(mean=0.0, std=1.0)
            self.fc2.weight.normal_(mean=0.0, std=1.0)
            self.fc1.weight.mul_(init_scale)
            self.fc2.weight.mul_(init_scale)
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.fc2(h)


# ---------------------- train/eval ----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.size(0))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)

    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    ap.add_argument("--batch_size", type=int, default=128)

    # optimizee definition (fairness knobs)
    ap.add_argument("--hidden_dim", type=int, default=20)
    ap.add_argument("--init_scale", type=float, default=1e-3)

    # training mode
    ap.add_argument("--optim_steps", type=int, default=100, help="Step-based training (recommended for fair compare to L2L).")
    ap.add_argument("--eval_every", type=int, default=10, help="Evaluate+log every N steps.")
    ap.add_argument("--epochs", type=int, default=0, help="If >0 and --optim_steps <=0, use epoch-based training.")

    # optimizer
    ap.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.0)

    # logging
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-online")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default="mnist1d_mlp_fair_baseline")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] device = {device}")

    # data
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = to_N40(xtr).astype(np.float32)
    xte = to_N40(xte).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, mean, std = load_split_and_norm(Path(args.preprocess_dir), args.data_seed)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)

    x_train = apply_norm(xtr[train_idx], mean, std)
    y_train = ytr[train_idx]
    x_val = apply_norm(xtr[val_idx], mean, std)
    y_val = ytr[val_idx]
    x_test = apply_norm(xte, mean, std)
    y_test = yte

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=512,
        shuffle=False,
    )

    # model
    model = MLPOptimizeeBaseline(hidden_dim=args.hidden_dim, init_scale=args.init_scale).to(device)

    # optimizer
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    # run dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"fair_mlp_{args.opt}_seed{args.seed}_data{args.data_seed}_"
        f"steps{args.optim_steps}_bs{args.batch_size}_lr{args.lr}_"
        f"hd{args.hidden_dim}_inits{args.init_scale}_{ts}"
    )
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    # log files
    train_log_path = run_dir / "train_log.csv"
    curve_path = run_dir / "curve.csv"
    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "elapsed_sec",
            "train_loss", "train_acc",
            "val_loss", "val_acc",
            "test_loss", "test_acc",
            "best_val_acc_so_far",
        ])
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])

    # wandb
    wb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was passed.")
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    # train
    best_val_acc = -1.0
    best_step = -1
    start_time = time.time()

    if args.optim_steps and args.optim_steps > 0:
        gen = infinite_loader(train_loader)
        model.train()

        for step in range(1, args.optim_steps + 1):
            x, y = next(gen)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            with open(curve_path, "a", newline="") as f:
                csv.writer(f).writerow([step, float(loss.item())])

            if (step % args.eval_every) == 0 or step == args.optim_steps:
                tr_loss, tr_acc = evaluate(model, train_loader, device)
                va_loss, va_acc = evaluate(model, val_loader, device)
                te_loss, te_acc = evaluate(model, test_loader, device)

                elapsed = time.time() - start_time

                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    best_step = step
                    torch.save(model.state_dict(), run_dir / "best.pt")

                print(
                    f"[STEP {step:04d}] "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                    f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                    f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} "
                    f"(best_val_acc={best_val_acc:.4f} @ step={best_step})"
                )

                with open(train_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        step, f"{elapsed:.3f}",
                        f"{tr_loss:.8f}", f"{tr_acc:.6f}",
                        f"{va_loss:.8f}", f"{va_acc:.6f}",
                        f"{te_loss:.8f}", f"{te_acc:.6f}",
                        f"{best_val_acc:.6f}",
                    ])

                if wb_run is not None:
                    wandb.log({
                        "step": step,
                        "time/elapsed_sec": elapsed,
                        "train/loss": tr_loss,
                        "train/acc": tr_acc,
                        "val/loss": va_loss,
                        "val/acc": va_acc,
                        "test/loss": te_loss,
                        "test/acc": te_acc,
                        "best/val_acc": best_val_acc,
                    })

    else:
        # epoch mode (optional)
        if args.epochs <= 0:
            raise ValueError("If --optim_steps <= 0, please set --epochs > 0.")
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

            step = epoch + 1
            tr_loss, tr_acc = evaluate(model, train_loader, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            te_loss, te_acc = evaluate(model, test_loader, device)
            elapsed = time.time() - start_time

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_step = step
                torch.save(model.state_dict(), run_dir / "best.pt")

            print(
                f"[EPOCH {epoch:03d}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} "
                f"(best_val_acc={best_val_acc:.4f} @ epoch={best_step})"
            )

            with open(train_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, f"{elapsed:.3f}",
                    f"{tr_loss:.8f}", f"{tr_acc:.6f}",
                    f"{va_loss:.8f}", f"{va_acc:.6f}",
                    f"{te_loss:.8f}", f"{te_acc:.6f}",
                    f"{best_val_acc:.6f}",
                ])

            if wb_run is not None:
                wandb.log({
                    "epoch": epoch,
                    "time/elapsed_sec": elapsed,
                    "train/loss": tr_loss,
                    "train/acc": tr_acc,
                    "val/loss": va_loss,
                    "val/acc": va_acc,
                    "test/loss": te_loss,
                    "test/acc": te_acc,
                    "best/val_acc": best_val_acc,
                })

    # final summary (also compute test acc at best val)
    best_path = run_dir / "best.pt"
    test_acc_at_best_val = None
    test_loss_at_best_val = None
    if best_path.exists():
        model_best = MLPOptimizeeBaseline(hidden_dim=args.hidden_dim, init_scale=args.init_scale).to(device)
        model_best.load_state_dict(torch.load(best_path, map_location=device))
        test_loss_at_best_val, test_acc_at_best_val = evaluate(model_best, test_loader, device)

    result = {
        "method": "mnist1d_fair_baseline",
        "optimizee": f"MLP(40->{args.hidden_dim}(sigmoid)->10)",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "opt": str(args.opt),
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "batch_size": int(args.batch_size),
        "optim_steps": int(args.optim_steps),
        "epochs": int(args.epochs),
        "init_scale": float(args.init_scale),
        "hidden_dim": int(args.hidden_dim),
        "best_val_acc": float(best_val_acc),
        "best_step_or_epoch": int(best_step),
        "test_acc_at_best_val": (None if test_acc_at_best_val is None else float(test_acc_at_best_val)),
        "test_loss_at_best_val": (None if test_loss_at_best_val is None else float(test_loss_at_best_val)),
        "run_dir": str(run_dir),
        "checkpoint": str(best_path),
        "artifacts": {
            "train_log": str(train_log_path),
            "curve": str(curve_path),
        },
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[RESULT] best_val_acc={best_val_acc:.4f} @ {best_step}")
    if test_acc_at_best_val is not None:
        print(f"[RESULT] test_acc_at_best_val={test_acc_at_best_val:.4f} test_loss_at_best_val={test_loss_at_best_val:.4f}")

    if wb_run is not None:
        wandb.summary["best_val_acc"] = float(best_val_acc)
        wandb.summary["best_step_or_epoch"] = int(best_step)
        if test_acc_at_best_val is not None:
            wandb.summary["test_acc_at_best_val"] = float(test_acc_at_best_val)
            wandb.summary["test_loss_at_best_val"] = float(test_loss_at_best_val)
        # upload artifacts
        for p in [train_log_path, curve_path, run_dir / "result.json", best_path]:
            if p.exists():
                wandb.save(str(p))
        wandb.finish()


if __name__ == "__main__":
    main()

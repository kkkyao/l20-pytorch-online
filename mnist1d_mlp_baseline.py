#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully-aligned MLP baseline on MNIST-1D.

Alignment guarantees:
  - Same split / norm artifacts
  - loader_utils RNG semantics
  - Local logs: curve.csv / train_log.csv / time_log.csv / result.json
  - WandB logging aligned with Conv1D / Conv2D / ResNet18 baselines
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from data_mnist1d import load_mnist1d
from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader


# ---------------------- Utilities ----------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_N40(x, length: int = 40) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected shape: {x.shape}")


def apply_norm_np(x, mean, std, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((x - mean[None, :]) / (std[None, :] + eps)).astype(np.float32)


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = preprocess_dir / f"seed_{data_seed}"
    with open(pdir / "split.json") as f:
        split = json.load(f)
    with open(pdir / "norm.json") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model ----------------------
class MLPBaseline(nn.Module):
    def __init__(self, input_len=40, hidden_dim=128, num_layers=5):
        super().__init__()
        layers = []
        in_dim = input_len
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        return self.out(self.mlp(x))


# ---------------------- Loggers ----------------------
class TimeLogger:
    """
    STRICT time_log.csv
    Schema (hard contract):
      elapsed_sec,train_loss,train_acc,test_loss,test_acc
    """
    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.start_time = None

        with open(self.out_csv, "w") as f:
            f.write("elapsed_sec,train_loss,train_acc,test_loss,test_acc\n")

    def start(self):
        self.start_time = time.time()

    def log_epoch(
        self,
        train_loss, train_acc,
        test_loss, test_acc,
    ):
        elapsed = time.time() - self.start_time
        rec = {
            "elapsed_sec": elapsed,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        with open(self.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            writer.writerow(rec)


class BatchLossLogger:
    def __init__(self, csv_path: Path, flush_every: int = 200):
        self.csv_path = csv_path
        with open(self.csv_path, "w") as f:
            f.write("iter,epoch,loss\n")
        self.iter = 0
        self.rows = []
        self.flush_every = flush_every

    def log(self, epoch: int, loss_value: float):
        self.rows.append([self.iter, epoch, loss_value])
        self.iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def _flush(self):
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
        self.rows.clear()

    def close(self):
        self._flush()


# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, batch_logger, epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_logger.log(epoch, loss.item())

        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return loss_sum / total, correct / total


# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    p.add_argument("--out_root", type=str, default="runs")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="leyao-li-epfl")
    p.add_argument("--wandb_group", type=str, required=False)
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = to_N40(xtr).astype(np.float32)
    xte = to_N40(xte).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean, std = norm["mean"], norm["std"]

    x_train = apply_norm_np(xtr[split["train_idx"]], mean, std)
    x_val = apply_norm_np(xtr[split["val_idx"]], mean, std)
    x_test = apply_norm_np(xte, mean, std)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(ytr[split["train_idx"]]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(ytr[split["val_idx"]]).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(yte).long(),
    )

    cfg = LoaderCfg(batch_size=args.batch_size, num_workers=0)
    train_loader, val_loader = make_train_val_loaders(
        train_ds, val_ds, cfg,
        seed=args.seed,
        train_shuffle=True,
        val_shuffle=True,
        train_drop_last=True,
        val_drop_last=True,
    )
    test_loader = make_eval_loader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # ---------------- model ----------------
    model = MLPBaseline(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
        )

    criterion = nn.CrossEntropyLoss()

    # ---------------- logging ----------------
    run_dir = (
        Path(args.out_root)
        / f"baseline_mlp_{args.opt}_seed{args.seed}_dataseed{args.data_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    batch_logger = BatchLossLogger(run_dir / "curve.csv")

    # STRICT train_log.csv
    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")

    # ---------------- wandb ----------------
    wandb_run = None
    if args.wandb:
        import wandb

        run_name = (
            args.wandb_run_name
            or f"MNIST-1D_MLP_baseline_{args.opt}_seed{args.seed}"
        )

        wandb_cfg = dict(vars(args))
        wandb_cfg.update({
            "dataset": "MNIST-1D",
            "backbone": "MLP",
            "method": "baseline",
        })

        wandb_run = wandb.init(
            project="l2o-online(new1)",
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config=wandb_cfg,
        )

    time_logger.start()

    # ---------------- training ----------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            batch_logger, epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        time_logger.log_epoch(
            train_loss, train_acc,
            test_loss, test_acc,
        )

        with open(train_log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_loss:.8f},{train_acc:.6f},"
                f"{test_loss:.8f},{test_acc:.6f}\n"
            )

        if wandb_run:
            import wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    batch_logger.close()

    final_test_loss, final_test_acc = evaluate(
        model, test_loader, criterion, device
    )

    result = {
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    if wandb_run:
        import wandb
        wandb.log(result)
        wandb.finish()


if __name__ == "__main__":
    main()

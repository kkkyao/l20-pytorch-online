#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP baseline on MNIST-1D with wall-clock time logging (PyTorch version).

Alignment:
  - Use the fixed split produced by preprocess:
        artifacts/preprocess/seed_{data_seed}/split.json
  - Use the same per-position normalization statistics:
        artifacts/preprocess/seed_{data_seed}/norm.json (mean/std)
  - Input is a length-40 vector (shape (40,)), MLP has 5 hidden layers.
  - Logs:
      (1) Per-batch training loss          -> curve.csv
      (2) Per-epoch train/val/test metrics -> train_log.csv
      (3) Per-epoch wall-clock time & perf -> time_log.csv
      (4) Final test_acc/test_loss         -> result.json
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
from torch.utils.data import TensorDataset, DataLoader

from data_mnist1d import load_mnist1d


# ---------------------- Utility: shape normalization ----------------------
def to_N40(x, length: int = 40) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected shape for x: {x.shape}")


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = preprocess_dir / f"seed_{data_seed}"
    with open(pdir / "split.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(pdir / "norm.json", "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model definition ----------------------
class MLPBaseline(nn.Module):
    def __init__(self, input_len: int = 40, hidden_dim: int = 128, num_layers: int = 5):
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
        x = self.mlp(x)
        return self.out(x)


# ---------------------- Logging helpers ----------------------
class TimeLogger:
    """
    epoch, elapsed_sec,
    train_loss, val_loss, train_acc, val_acc,
    test_loss, test_acc
    """

    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.start_time = None

    def on_train_begin(self):
        self.start_time = time.time()

    # [MOD] add test_loss, test_acc
    def on_epoch_end(
        self,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        test_loss,   # [ADD]
        test_acc,    # [ADD]
    ):
        elapsed = time.time() - self.start_time
        rec = {
            "epoch": int(epoch + 1),
            "elapsed_sec": float(elapsed),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        write_header = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(rec)


class MiniBatchLossLogger:
    def __init__(self, out_csv: Path, flush_every: int = 100):
        self.out_csv = out_csv
        with open(self.out_csv, "w") as f:
            f.write("step,epoch,loss\n")
        self.step = 0
        self.epoch = 0
        self.rows = []
        self.flush_every = flush_every

    def on_epoch_begin(self, epoch: int):
        self.epoch = epoch

    def on_train_batch_end(self, loss_value: float):
        self.rows.append([self.step, self.epoch, loss_value])
        self.step += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def on_train_end(self):
        if self.rows:
            self._flush()

    def _flush(self):
        with open(self.out_csv, "a") as f:
            for r in self.rows:
                f.write(",".join(map(str, r)) + "\n")
        self.rows.clear()


class CSVLogger:
    """
    epoch, accuracy, loss, val_accuracy, val_loss, test_accuracy, test_loss
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._initialized = False

    # [MOD] add test metrics
    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        test_loss,   # [ADD]
        test_acc,    # [ADD]
    ):
        rec = {
            "epoch": int(epoch),
            "accuracy": float(train_acc),
            "loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        }
        write_header = not self._initialized
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(rec)
        self._initialized = True


# ---------------------- Training / evaluation helpers ----------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, batch_logger=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if batch_logger:
            batch_logger.on_train_batch_end(loss.item())

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    p.add_argument("--lr", type=float, default=0.10)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mnist1d_mlp_baseline")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = to_N40(xtr).astype(np.float32)
    xte = to_N40(xte).astype(np.float32)

    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean, std = np.array(norm["mean"]), np.array(norm["std"])

    x_train = apply_norm_np(xtr[split["train_idx"]], mean, std)
    x_val = apply_norm_np(xtr[split["val_idx"]], mean, std)
    x_test = apply_norm_np(xte, mean, std)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(ytr[split["train_idx"]])),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(ytr[split["val_idx"]])),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(yte)),
        batch_size=args.batch_size
    )

    model = MLPBaseline().to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3, momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()

    run_dir = Path("runs") / f"baseline_mlp_{args.opt}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    csv_logger = CSVLogger(run_dir / "train_log.csv")
    batch_logger = MiniBatchLossLogger(run_dir / "curve.csv")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"baseline_mlp_{args.opt}_seed{args.seed}",
        )

    time_logger.on_train_begin()

    for epoch in range(args.epochs):
        batch_logger.on_epoch_begin(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, batch_logger
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # [ADD] per-epoch test evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        time_logger.on_epoch_end(
            epoch,
            train_loss, val_loss, train_acc, val_acc,
            test_loss, test_acc,
        )
        csv_logger.log_epoch(
            epoch,
            train_loss, val_loss, train_acc, val_acc,
            test_loss, test_acc,
        )

        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })

    batch_logger.on_train_end()

    # -------- final test (unchanged semantics) --------
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    with open(run_dir / "result.json", "w") as f:
        json.dump({
            "test_loss": final_test_loss,
            "test_acc": final_test_acc,
        }, f, indent=2)

    if wandb_run:
        import wandb
        wandb.log({
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
        })
        wandb.finish()


if __name__ == "__main__":
    main()

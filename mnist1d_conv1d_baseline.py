#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (Conv1D) on MNIST-1D with full alignment:
- loader_utils RNG semantics
- local CSV logging (curve / train_log / time_log)
- wandb logging aligned with all baselines / learned runs
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
from utils.plot_exporter import export_plot_files


# ---------------------- Utilities ----------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
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


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
    return (x - mean[None, :]) / (std[None, :] + eps)


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = preprocess_dir / f"seed_{data_seed}"
    with open(pdir / "split.json") as f:
        split = json.load(f)
    with open(pdir / "norm.json") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model ----------------------
class Conv1DMNIST1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------- Loggers ----------------------
class TimeLogger:
    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, val_loss, test_loss,
                  train_acc, val_acc, test_acc):
        rec = {
            "epoch": epoch,
            "elapsed_sec": time.time() - self.start_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }
        write_header = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(rec)


class BatchLossLogger:
    def __init__(self, out_csv: Path, flush_every: int = 200):
        self.out_csv = out_csv
        with open(self.out_csv, "w") as f:
            f.write("iter,epoch,loss\n")
        self.iter = 0
        self.rows = []
        self.flush_every = flush_every

    def log(self, epoch, loss):
        self.rows.append([self.iter, epoch, loss])
        self.iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def _flush(self):
        with open(self.out_csv, "a") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
        self.rows.clear()

    def close(self):
        self._flush()


class EpochLogger:
    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.initialized = False

    def log(self, epoch, train_loss, val_loss, test_loss,
            train_acc, val_acc, test_acc):
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        write_header = not self.initialized
        with open(self.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(rec)
        self.initialized = True


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

        batch_logger.log(epoch, float(loss.item()))
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

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
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    ap.add_argument("--out_root", type=str, default="runs")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)
    xtr = to_N40(xtr).astype(np.float32)
    xte = to_N40(xte).astype(np.float32)

    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    x_train = apply_norm_np(xtr[split["train_idx"]], mean, std)[..., None]
    x_val = apply_norm_np(xtr[split["val_idx"]], mean, std)[..., None]
    x_test = apply_norm_np(xte, mean, std)[..., None]

    def to_tensor(x):
        return torch.from_numpy(x).permute(0, 2, 1).contiguous()

    train_ds = TensorDataset(
        to_tensor(x_train), torch.from_numpy(ytr[split["train_idx"]]).long()
    )
    val_ds = TensorDataset(
        to_tensor(x_val), torch.from_numpy(ytr[split["val_idx"]]).long()
    )
    test_ds = TensorDataset(
        to_tensor(x_test), torch.from_numpy(yte).long()
    )

    cfg = LoaderCfg(batch_size=args.bs, num_workers=0)
    train_loader, val_loader = make_train_val_loaders(
        train_ds, val_ds, cfg,
        seed=args.seed,
        train_shuffle=True,
        val_shuffle=False,
        train_drop_last=True,
        val_drop_last=False,
    )
    test_loader = make_eval_loader(test_ds, batch_size=args.bs, num_workers=0)

    # ---------------- model ----------------
    model = Conv1DMNIST1D().to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    run_dir = (
        Path(args.out_root)
        / f"baseline_conv1d_{args.opt}_seed{args.seed}_dataseed{args.data_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    batch_logger = BatchLossLogger(run_dir / "curve.csv")
    epoch_logger = EpochLogger(run_dir / "train_log.csv")

    # ---------------- wandb (关键修复点) ----------------
    wandb_run = None
    if args.wandb:
        import wandb

        run_name = (
            args.wandb_run_name
            or f"MNIST-1D_Conv1D_baseline_{args.opt}_seed{args.seed}"
        )

        wandb_cfg = dict(vars(args))
        wandb_cfg.update({
            "dataset": "MNIST-1D",
            "backbone": "Conv1D",
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

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, batch_logger, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        time_logger.log_epoch(
            epoch, train_loss, val_loss, test_loss,
            train_acc, val_acc, test_acc
        )
        epoch_logger.log(
            epoch, train_loss, val_loss, test_loss,
            train_acc, val_acc, test_acc
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

    batch_logger.close()

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    with open(run_dir / "result.json", "w") as f:
        json.dump(
            {
                "final_test_loss": final_test_loss,
                "final_test_acc": final_test_acc,
            },
            f,
            indent=2,
        )

    if wandb_run:
        import wandb
        wandb.log(
            {
                "final_test_loss": final_test_loss,
                "final_test_acc": final_test_acc,
            }
        )
        wandb.finish()
    # ---------------- export paper-level plot files ----------------
    export_plot_files(
        run_dir=run_dir,
        dataset="mnist1d",
        model="conv1d",
        method="baseline",
        seed=args.seed,
        data_seed=args.data_seed,
        epochs=args.epochs,
        optimizer=args.opt,
    )


if __name__ == "__main__":
    main()

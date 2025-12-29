#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (Conv2D) on CIFAR-10 in PyTorch with wall-clock logging,
aligned with F1/F2/F3 and loader_utils RNG semantics.
"""

import argparse
import csv
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torchvision
from torchvision import transforms

from loader_utils import (
    LoaderCfg,
    make_train_val_loaders,
    make_eval_loader,
)

# ---------------------- Utilities ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = preprocess_dir / f"seed_{data_seed}"
    with open(pdir / "split.json") as f:
        split = json.load(f)
    with open(pdir / "norm.json") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model ----------------------
class Conv2DBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------- Loggers ----------------------
class TimeLogger:
    """
    epoch, elapsed_sec,
    train_loss, val_loss, test_loss,
    train_acc, val_acc, test_acc
    """

    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch,
        train_loss, val_loss, test_loss,
        train_acc, val_acc, test_acc,
    ):
        elapsed = time.time() - self.start_time
        rec = {
            "epoch": epoch + 1,
            "elapsed_sec": elapsed,
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


def append_train_log(csv_path: Path, epoch,
                     train_loss, train_acc,
                     val_loss, val_acc,
                     test_loss, test_acc):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "loss", "accuracy",
                "val_loss", "val_accuracy",
                "test_loss", "test_accuracy",
            ])
        writer.writerow([
            epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            test_loss, test_acc,
        ])


# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, device, loader, criterion, optimizer, batch_logger, epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        batch_logger.log(epoch, loss.item())
        loss_sum += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def evaluate(model, device, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/cifar10_conv2d_preprocess",
    )

    # NEW: output root (to move local saving path)
    ap.add_argument("--out_root", type=str, default="runs")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="cifar10_conv2d_baseline")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)

    # NEW: wandb run name override (avoid duplicate names across opts)
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean, std = norm["mean"], norm["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = torchvision.datasets.CIFAR10(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "data", train=False, download=True, transform=transform
    )

    train_dataset = Subset(train_full, split["train_idx"])
    val_dataset = Subset(train_full, split["val_idx"])

    # ---------------- loaders (via loader_utils) ----------------
    cfg = LoaderCfg(
        batch_size=args.bs,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader, val_loader = make_train_val_loaders(
        train_dataset,
        val_dataset,
        cfg,
        seed=args.seed,
        train_shuffle=True,
        val_shuffle=False,      # baseline: epoch-wise validation
        train_drop_last=True,
        val_drop_last=False,
    )

    test_loader = make_eval_loader(
        test_dataset,
        batch_size=args.bs,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------- model / optim ----------------
    model = Conv2DBaseline().to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    # ---------------- logging ----------------
    run_dir = Path(args.out_root) / f"baseline_conv2d_{args.opt}_seed{args.seed}_dataseed{args.data_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    batch_logger = BatchLossLogger(run_dir / "curve.csv")
    train_log_csv = run_dir / "train_log.csv"

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"baseline_conv2d_{args.opt}_seed{args.seed}",
        )

    time_logger.start()

    # ---------------- training ----------------
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer, batch_logger, epoch
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        time_logger.log_epoch(
            epoch,
            train_loss, val_loss, test_loss,
            train_acc, val_acc, test_acc,
        )
        append_train_log(
            train_log_csv,
            epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            test_loss, test_acc,
        )

        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}"
        )

    batch_logger.close()

    # ---------------- final test ----------------
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

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

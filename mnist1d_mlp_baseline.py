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
      (2) Per-epoch train/val metrics      -> train_log.csv
      (3) Per-epoch wall-clock time & perf -> time_log.csv
      (4) Final test_acc/test_loss         -> result.json

Optional:
  - Log metrics and artifacts to Weights & Biases (wandb) with --wandb flag.
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
    """
    Convert input to shape [N, length], supporting:
      - [N, length]
      - [N, length, 1]
      - [N, 1, length]
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:   # [N,40,1] -> [N,40]
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:   # [N,1,40] -> [N,40]
            return x[:, 0, :]
    raise AssertionError(
        f"Unexpected shape for x: {x.shape}, expect [N,{length}] or [N,{length},1]/[N,1,{length}]"
    )


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-position standardization: (x - mean) / std, x: [N,40]."""
    return (x - mean[None, :]) / std[None, :]


def load_artifacts(preprocess_dir: Path, data_seed: int):
    """
    Load the same split and normalization stats as F1/F2/F3:
      - split.json: train_idx / val_idx
      - norm.json:  mean / std
    """
    pdir = preprocess_dir / f"seed_{data_seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifacts under: {pdir}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model definition ----------------------
class MLPBaseline(nn.Module):
    """
    MLP with input length-40 vector and 5 hidden layers of width 128 (ReLU),
    followed by a 10-way linear classifier (logits).
    """

    def __init__(self, input_len: int = 40, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        layers_list = []
        in_dim = input_len
        for _ in range(num_layers):
            layers_list.append(nn.Linear(in_dim, hidden_dim))
            layers_list.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers_list)
        self.out = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 40]
        x = self.mlp(x)
        logits = self.out(x)
        return logits


# ---------------------- Logging helpers ----------------------
class TimeLogger:
    """
    Log wall-clock time and epoch-level metrics to CSV.

    Columns:
      epoch, elapsed_sec, train_loss, val_loss, train_acc, val_acc
    """

    def __init__(self, out_csv: Path):
        self.out_csv = Path(out_csv)
        self.start_time = None

    def on_train_begin(self):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, train_loss, val_loss, train_acc, val_acc):
        if self.start_time is None:
            raise RuntimeError("TimeLogger.on_train_begin must be called before training.")
        elapsed = time.time() - self.start_time
        rec = {
            "epoch": int(epoch + 1),
            "elapsed_sec": float(elapsed),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }
        write_header = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(rec)


class MiniBatchLossLogger:
    """
    Log per-batch training loss to CSV.

    Columns:
      step, epoch, loss
    """

    def __init__(self, out_csv: Path, flush_every: int = 100):
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w") as f:
            f.write("step,epoch,loss\n")
        self.step = 0
        self.epoch = 0
        self.rows = []
        self.flush_every = flush_every

    def on_epoch_begin(self, epoch: int):
        self.epoch = int(epoch)

    def on_train_batch_end(self, loss_value: float):
        self.rows.append([self.step, self.epoch, float(loss_value)])
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
    Simple CSV logger roughly aligned with Keras CSVLogger.

    Columns (per epoch):
      epoch,accuracy,loss,val_accuracy,val_loss
    """

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self._initialized = False

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        rec = {
            "epoch": int(epoch),
            "accuracy": float(train_acc),
            "loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
        }
        write_header = not self._initialized and not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(rec)
        self._initialized = True


# ---------------------- Training / evaluation helpers ----------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_logger: MiniBatchLossLogger = None,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        if batch_logger is not None:
            batch_logger.on_train_batch_end(batch_loss)

        running_loss += batch_loss * targets.size(0)
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        running_total += targets.size(0)

    avg_loss = running_loss / running_total if running_total > 0 else float("nan")
    acc = running_correct / running_total if running_total > 0 else 0.0
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += targets.size(0)

    avg_loss = running_loss / running_total if running_total > 0 else float("nan")
    acc = running_correct / running_total if running_total > 0 else 0.0
    return avg_loss, acc


# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training seed (model init, shuffling, etc.)",
    )
    p.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="fixed data split seed (must match F1/F2/F3 preprocess)",
    )
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--opt",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "rmsprop"],
        help="optimizer for the MLP baseline",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.10,
        help="SGD learning rate (ignored for Adam/RMSprop)",
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="momentum for SGD/RMSprop",
    )
    p.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/preprocess",
        help="directory containing preprocess artifacts",
    )

    # W&B related arguments
    p.add_argument(
        "--wandb",
        action="store_true",
        help="enable logging to Weights & Biases (wandb)",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="mnist1d_mlp_baseline",
        help="W&B project name",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/user name (optional)",
    )
    p.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name (optional)",
    )

    args = p.parse_args()

    # Fix random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ========= Data loading (aligned with F1/F2/F3) =========
    # 1) Load MNIST-1D using data_seed (only depends on data_seed)
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)

    # 2) Normalize shapes to [N, 40] and cast to float32/int64
    xtr = to_N40(xtr, length=40).astype(np.float32)
    xte = to_N40(xte, length=40).astype(np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    # 3) Load fixed train/val split and normalization statistics
    split, norm = load_artifacts(Path(args.preprocess_dir), data_seed=args.data_seed)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    # 4) Build x_train/x_val/x_test and apply per-feature standardization
    x_train_raw, y_train = xtr[train_idx], ytr[train_idx]
    x_val_raw, y_val = xtr[val_idx], ytr[val_idx]
    x_test_raw, y_test = xte, yte

    x_train = apply_norm_np(x_train_raw, mean, std)   # [N,40]
    x_val = apply_norm_np(x_val_raw, mean, std)
    x_test = apply_norm_np(x_test_raw, mean, std)

    # Convert to PyTorch tensors
    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)

    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)
    y_test_t = torch.from_numpy(y_test)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ========= Build MLP model =========
    model = MLPBaseline(input_len=40).to(device)

    # Optimizer selection
    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=1e-3,
            momentum=args.momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.opt}")

    criterion = nn.CrossEntropyLoss()

    # ========= Logging directory and loggers =========
    run_dir = Path("runs") / f"baseline_mlp_{args.opt}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    csv_logger = CSVLogger(run_dir / "train_log.csv")
    batch_logger = MiniBatchLossLogger(run_dir / "curve.csv")

    # ========= Optional: Weights & Biases logging =========
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Please `pip install wandb` or "
                "run without the --wandb flag."
            ) from e

        run_name = (
            f"baseline_mlp_mnist1d_{args.opt}_seed{args.seed}_data{args.data_seed}"
        )

        wandb_config = {
            "method": "baseline_mlp_mnist1d",
            "dataset": "MNIST-1D",
            "optimizer": args.opt,
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "momentum": float(args.momentum),
            "preprocess_dir": str(args.preprocess_dir),
            "device": str(device),
        }

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config=wandb_config,
        )

    # ========= Training =========
    time_logger.on_train_begin()
    train_start = time.time()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        batch_logger.on_epoch_begin(epoch)

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_logger=batch_logger,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        time_logger.on_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )

        csv_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )

        print(
            f"  train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

    batch_logger.on_train_end()
    total_train_time = time.time() - train_start

    # ========= Test =========
    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"[MLP_BASELINE] opt={args.opt} seed={args.seed} data_seed={args.data_seed} "
        f"TestAcc={test_acc:.4f}, TestLoss={test_loss:.4f}"
    )
    print(f"[LOG] Time log saved to: {time_logger.out_csv.resolve()}")

    # ========= Save results locally =========
    result = {
        "method": f"baseline_mlp_{args.opt}",
        "optimizer": args.opt,
        "dataset": "MNIST-1D",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "train_elapsed_sec": float(total_train_time),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[LOG] Results saved to: {run_dir.resolve()}")

    # ========= Also log final metrics & artifacts to wandb (if enabled) =========
    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "train_elapsed_sec": float(total_train_time),
            }
        )

        # Try saving local CSV/JSON logs as artifacts
        try:
            wandb.save(str(run_dir / "time_log.csv"))
            wandb.save(str(run_dir / "curve.csv"))
            wandb.save(str(run_dir / "train_log.csv"))
            wandb.save(str(run_dir / "result.json"))
        except Exception as e:
            print(f"[W&B] Warning: failed to save local log files to wandb: {e}")

        wandb.finish()


if __name__ == "__main__":
    main()

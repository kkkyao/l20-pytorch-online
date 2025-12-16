#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP baseline on MNIST-1D with wall-clock time logging (PyTorch version).

Alignment:
  - Use the fixed split produced by preprocess:
        artifacts/preprocess/seed_{data_seed}/split.json
  - Use the same per-position normalization statistics:
        artifacts/preprocess/seed_{data_seed}/norm.json (mean/std)
  - Input is a length-40 vector (shape (40,)), MLP has configurable hidden layers.
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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_mnist1d import load_mnist1d

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ---------------------- Utils ----------------------
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
    raise AssertionError(f"Unexpected shape for x: {x.shape}")


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    IMPORTANT: return float32 to avoid torch Double/Float mismatch.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    out = (x - mean[None, :]) / (std[None, :] + eps)
    return out.astype(np.float32, copy=False)


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = preprocess_dir / f"seed_{data_seed}"
    with open(pdir / "split.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(pdir / "norm.json", "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- Model ----------------------
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def on_epoch_end(self, epoch, train_loss, val_loss, train_acc, val_acc, test_loss, test_acc):
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
    epoch, accuracy, loss, val_accuracy, val_loss, test_accuracy, test_loss
    """
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._initialized = False

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, test_loss, test_acc):
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


# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, batch_logger=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in dataloader:
        # IMPORTANT: force dtype
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if batch_logger:
            batch_logger.on_train_batch_end(float(loss.item()))

        bs = int(y.size(0))
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(1) == y).sum().item())
        total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in dataloader:
        # IMPORTANT: force dtype
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        logits = model(x)
        loss = criterion(logits, y)

        bs = int(y.size(0))
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(1) == y).sum().item())
        total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


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
    p.add_argument("--lr", type=float, default=0.10)
    p.add_argument("--momentum", type=float, default=0.0)

    p.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mnist1d_mlp_baseline")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=args.data_seed)

    xtr = to_N40(xtr).astype(np.float32, copy=False)
    xte = to_N40(xte).astype(np.float32, copy=False)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)

    x_train = apply_norm_np(xtr[train_idx], mean, std)
    x_val = apply_norm_np(xtr[val_idx], mean, std)
    x_test = apply_norm_np(xte, mean, std)

    # Ensure float32 in torch tensors
    x_train_t = torch.from_numpy(x_train).to(torch.float32)
    x_val_t = torch.from_numpy(x_val).to(torch.float32)
    x_test_t = torch.from_numpy(x_test).to(torch.float32)

    y_train_t = torch.from_numpy(ytr[train_idx]).to(torch.long)
    y_val_t = torch.from_numpy(ytr[val_idx]).to(torch.long)
    y_test_t = torch.from_numpy(yte).to(torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_t, y_test_t),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---------------- model ----------------
    model = MLPBaseline(input_len=40, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    # ---------------- run dir ----------------
    run_name = (
        args.wandb_run_name
        or f"baseline_mlp_{args.opt}_seed{args.seed}_data{args.data_seed}_"
           f"ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_hd{args.hidden_dim}_nl{args.num_layers}_"
           f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    time_logger = TimeLogger(run_dir / "time_log.csv")
    csv_logger = CSVLogger(run_dir / "train_log.csv")
    batch_logger = MiniBatchLossLogger(run_dir / "curve.csv")

    # ---------------- wandb ----------------
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config=vars(args),
        )

    # ---------------- train loop ----------------
    time_logger.on_train_begin()

    best_val_acc = -1.0
    best_epoch = -1
    best_test_acc_at_best_val = -1.0

    for epoch in range(args.epochs):
        batch_logger.on_epoch_begin(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, batch_logger
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        time_logger.on_epoch_end(
            epoch, train_loss, val_loss, train_acc, val_acc, test_loss, test_acc
        )
        csv_logger.log_epoch(
            epoch, train_loss, val_loss, train_acc, val_acc, test_loss, test_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            best_test_acc_at_best_val = float(test_acc)
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best/val_acc": best_val_acc,
                }
            )

    batch_logger.on_train_end()

    # -------- final test --------
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

    result = {
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "opt": str(args.opt),
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "test_acc_at_best_val": float(best_test_acc_at_best_val),
        "run_dir": str(run_dir),
        "artifacts": {
            "curve_csv": str(run_dir / "curve.csv"),
            "train_log_csv": str(run_dir / "train_log.csv"),
            "time_log_csv": str(run_dir / "time_log.csv"),
            "best_model_pt": str(run_dir / "best_model.pt"),
        },
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"[RESULT] final_test_acc={final_test_acc:.4f} final_test_loss={final_test_loss:.4f} "
        f"best_val_acc={best_val_acc:.4f} (epoch={best_epoch})"
    )

    if wandb_run is not None:
        wandb_run.summary["final_test_acc"] = float(final_test_acc)
        wandb_run.summary["final_test_loss"] = float(final_test_loss)
        wandb_run.summary["best_val_acc"] = float(best_val_acc)
        wandb_run.summary["best_epoch"] = int(best_epoch)
        wandb_run.summary["test_acc_at_best_val"] = float(best_test_acc_at_best_val)

        # Upload artifacts
        for pth in [
            run_dir / "curve.csv",
            run_dir / "train_log.csv",
            run_dir / "time_log.csv",
            run_dir / "result.json",
            run_dir / "best_model.pt",
        ]:
            if pth.exists():
                wandb.save(str(pth))

        wandb.finish()


if __name__ == "__main__":
    main()

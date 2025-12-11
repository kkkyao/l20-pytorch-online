#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (ResNet-18) on CIFAR-10 in PyTorch with wall-clock logging, aligned with F1/F2/F3.

Alignments:
  - Use preprocess/cifar10_conv2d_preprocess/seed_{data_seed}/
      split.json (train_idx/val_idx)
      norm.json  (mean[3], std[3])
  - Input: [N, 3, 32, 32]
  - Output logs in run_dir:
      curve.csv        (batch-level training loss vs iteration)
      time_log.csv     (epoch-level wall-clock time and metrics)
      train_log.csv    (epoch-level train/val metrics)
      result.json      (final test metrics and meta-info)

Optional:
  - Log metrics and artifacts to Weights & Biases (wandb) with --wandb flag.
  - Optionally save a warm-start checkpoint (--save_init_path / --save_init_epoch)
    that can be reused for meta-train.
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
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms


# ---------------------- Utilities: seeding & artifacts ----------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_artifacts(preprocess_dir: Path, data_seed: int):
    """
    Load split and normalization statistics from:
        preprocess_dir / f"seed_{data_seed}"/split.json
        preprocess_dir / f"seed_{data_seed}"/norm.json
    """
    pdir = preprocess_dir / f"seed_{data_seed}"
    split_path, norm_path = pdir / "split.json", pdir / "norm.json"
    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifacts under: {pdir}")
    with open(split_path, "r") as f:
        split = json.load(f)
    with open(norm_path, "r") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- ResNet-18 backbone for CIFAR-10 ----------------------
class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 backbone adapted for CIFAR-10 (32x32 inputs).

    Modifications:
      - conv1: 7x7, stride=2 -> 3x3, stride=1, padding=1
      - maxpool: replaced by Identity (no pooling at the first stage)
      - fc: output dimension changed to 10 classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        backbone = torchvision.models.resnet18(pretrained=False)

        # Adapt first conv layer for 32x32 CIFAR-10 images
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove the initial 3x3 max pooling (too aggressive for 32x32)
        backbone.maxpool = nn.Identity()

        # Replace final classification layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------- Loggers ----------------------
class TimeLogger:
    """
    Log wall-clock time and epoch-level metrics to a CSV file.

    Columns:
      epoch, elapsed_sec, train_loss, val_loss, train_acc, val_acc
    """

    def __init__(self, out_csv: Path):
        self.out_csv = Path(out_csv)
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float):
        if self.start_time is None:
            return
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


class BatchLossLogger:
    """
    Log per-batch training loss to a CSV file with columns:
      iter, epoch, loss
    """

    def __init__(self, csv_path: Path, flush_every: int = 200):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(self.csv_path, "w", newline="") as f:
            f.write("iter,epoch,loss\n")
        self.iter = 0
        self.flush_every = flush_every
        self.rows = []

    def log(self, epoch: int, loss_value: float):
        self.rows.append([self.iter, epoch, float(loss_value)])
        self.iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for r in self.rows:
                writer.writerow(r)
        self.rows = []

    def close(self):
        self._flush()


def append_train_log(csv_path: Path, epoch: int,
                     train_loss: float, train_acc: float,
                     val_loss: float, val_acc: float):
    """
    Append one epoch of train/val metrics to train_log.csv.

    Columns:
      epoch,loss,accuracy,val_loss,val_accuracy

    This mimics Keras CSVLogger output format.
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        writer.writerow([
            int(epoch),
            float(train_loss),
            float(train_acc),
            float(val_loss),
            float(val_acc),
        ])


# ---------------------- Training & evaluation helpers ----------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer,
                    batch_logger: BatchLossLogger, epoch_idx: int):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += batch_size

        batch_logger.log(epoch_idx, loss.item())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            _, preds = outputs.max(1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ---------------------- Main training function ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", type=str, default="sgd",
                    choices=["sgd", "adam", "rmsprop"],
                    help="which optimizer to use for the baseline")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed (model init, shuffling, etc.)")
    ap.add_argument("--data_seed", type=int, default=42,
                    help="fixed data split seed (must match preprocess)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--preprocess_dir", type=str,
                    default="artifacts/cifar10_conv2d_preprocess")

    # ---- warm-start checkpoint saving options ----
    ap.add_argument(
        "--save_init_path",
        type=str,
        default=None,
        help=(
            "If not None, save model.state_dict() to this path at a chosen epoch. "
            "Useful for warm-start meta-train initialization."
        ),
    )
    ap.add_argument(
        "--save_init_epoch",
        type=int,
        default=None,
        help=(
            "1-based epoch index at which to save the checkpoint specified by "
            "--save_init_path. If None, the checkpoint is saved at the last epoch."
        ),
    )

    # W&B related arguments
    ap.add_argument("--wandb", action="store_true",
                    help="enable logging to Weights & Biases (wandb)")
    ap.add_argument("--wandb_project", type=str,
                    default="cifar10_resnet18_baseline",
                    help="W&B project name")
    ap.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity/user name (optional)")
    ap.add_argument("--wandb_group", type=str, default=None,
                    help="W&B group name (optional)")

    args = ap.parse_args()

    # Set random seeds
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ========= 1. Load preprocess artifacts (split & norm) =========
    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

    mean = [float(m) for m in norm["mean"]]  # length 3
    std = [float(s) for s in norm["std"]]    # length 3

    # ========= 2. Build CIFAR-10 datasets & loaders =========
    transform = transforms.Compose([
        transforms.ToTensor(),               # [0,1], shape [C,H,W]
        transforms.Normalize(mean=mean, std=std),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(train_full, val_idx)

    # DataLoaders
    num_workers = 4
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ========= 3. Build ResNet-18 baseline model =========
    model = ResNet18CIFAR().to(device)

    if args.opt == "sgd":
        # 为了和 conv2d baseline 对齐，保持相同的 lr / momentum
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    # ========= 4. Logging directory =========
    run_dir = Path("runs_resnet18") / f"baseline_resnet18_{args.opt}_seed{args.seed}_dataseed{args.data_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    time_logger = TimeLogger(run_dir / "time_log.csv")
    batch_logger = BatchLossLogger(run_dir / "curve.csv")
    train_log_csv = run_dir / "train_log.csv"

    # ========= 5. Optional: Weights & Biases logging =========
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Please `pip install wandb` or "
                "run without the --wandb flag."
            ) from e

        run_name = f"baseline_resnet18_cifar10_{args.opt}_seed{args.seed}_data{args.data_seed}"

        wandb_config = {
            "method": "baseline_resnet18_cifar10_pytorch",
            "dataset": "CIFAR-10",
            "optimizer": args.opt,
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.bs),
            "preprocess_dir": str(args.preprocess_dir),
        }

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config=wandb_config,
        )

    # ========= 6. Training loop =========
    time_logger.start()
    train_start_wall = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer, batch_logger, epoch
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        time_logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc)
        append_train_log(train_log_csv, epoch, train_loss, train_acc, val_loss, val_acc)

        print(
            f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if wandb_run is not None:
            import wandb  # type: ignore
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

        # ---- optionally save a warm-start checkpoint for meta-train ----
        if args.save_init_path is not None:
            if args.save_init_epoch is None:
                should_save = (epoch + 1 == args.epochs)
            else:
                should_save = (epoch + 1 == args.save_init_epoch)

            if should_save:
                ckpt_path = Path(args.save_init_path)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"[BASELINE-RESNET18] Saved warm-start init to: {ckpt_path}")

    batch_logger.close()
    total_train_time = time.time() - train_start_wall

    # ========= 7. Test =========
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(
        f"[BASELINE RESNET18] opt={args.opt} seed={args.seed} data_seed={args.data_seed} "
        f"TestAcc={test_acc:.4f}, TestLoss={test_loss:.4f}"
    )
    print(f"[LOG] Time log saved to: {time_logger.out_csv.resolve()}")

    # ========= 8. Save results locally =========
    result = {
        "method": f"baseline_resnet18_{args.opt}_pytorch",
        "dataset": "CIFAR-10 ResNet18",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "train_elapsed_sec": float(total_train_time),
        "run_dir": str(run_dir),
        "save_init_path": str(args.save_init_path) if args.save_init_path is not None else None,
        "save_init_epoch": int(args.save_init_epoch) if args.save_init_epoch is not None else None,
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # ========= 9. Also log final metrics & artifacts to wandb (if enabled) =========
    if wandb_run is not None:
        import wandb  # type: ignore

        wandb.log(
            {
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "train_elapsed_sec": float(total_train_time),
            }
        )

        try:
            wandb.save(str(run_dir / "time_log.csv"))
            wandb.save(str(run_dir / "curve.csv"))
            wandb.save(str(run_dir / "train_log.csv"))
            wandb.save(str(run_dir / "result.json"))
        except Exception as e:  # noqa: BLE001
            print(f"[W&B] Warning: failed to save local log files to wandb: {e}")

        wandb.finish()


if __name__ == "__main__":
    main()

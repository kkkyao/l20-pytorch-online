#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-100 ResNet-50 Baseline (SGD/Adam/RMSprop) with checklist-aligned logging.

Key:
  - Uses preprocess artifacts (split.json / norm.json) from:
      artifacts/cifar100_resnet50_preprocess/seed_<data_seed>/
    where norm.json contains per-channel mean/std computed on TRAIN SUBSET (Strategy B).
  - Logs BOTH val and test metrics:
      train_log.csv : epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc
      time_log.csv  : elapsed_sec,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc
  - Also writes batch curve:
      curve.csv : iter,epoch,loss
  - Writes result.json with final metrics.

Backbone:
  - torchvision resnet50 with CIFAR stem (3x3 stride=1) and no maxpool.
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

from loader_utils import LoaderCfg, make_train_val_loaders, make_eval_loader

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ---------------------- Utilities ----------------------
def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_artifacts(preprocess_dir: Path, data_seed: int):
    pdir = Path(preprocess_dir) / f"seed_{int(data_seed)}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifacts under: {pdir}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


# ---------------------- ResNet-50 backbone (CIFAR stem) ----------------------
class ResNet50CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        backbone = torchvision.models.resnet50(pretrained=False)

        # CIFAR-style stem
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ---------------------- Loggers ----------------------
class TimeLogger:
    """
    Checklist-aligned time-based logger.
    Schema:
      elapsed_sec,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc
    """
    def __init__(self, out_csv: Path):
        self.out_csv = Path(out_csv)
        self.start_time = None
        with open(self.out_csv, "w", encoding="utf-8") as f:
            f.write("elapsed_sec,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n")

    def start(self):
        self.start_time = time.time()

    def log_epoch(self, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
        if self.start_time is None:
            raise RuntimeError("TimeLogger.start() must be called before log_epoch().")
        elapsed = time.time() - self.start_time
        with open(self.out_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{elapsed:.3f},"
                f"{train_loss:.8f},{train_acc:.6f},"
                f"{val_loss:.8f},{val_acc:.6f},"
                f"{test_loss:.8f},{test_acc:.6f}\n"
            )


class BatchLossLogger:
    """
    Batch-level training loss curve (auxiliary file).
    Schema:
      iter,epoch,loss
    """
    def __init__(self, csv_path: Path, flush_every: int = 200):
        self.csv_path = Path(csv_path)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("iter,epoch,loss\n")
        self.iter = 0
        self.rows = []
        self.flush_every = int(flush_every)

    def log(self, epoch: int, loss_value: float):
        self.rows.append([self.iter, int(epoch), float(loss_value)])
        self.iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
        self.rows.clear()

    def close(self):
        self._flush()


# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, device, loader, criterion, optimizer, batch_logger, epoch: int):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        batch_logger.log(epoch, float(loss.item()))

        bs = int(y.size(0))
        loss_sum += float(loss.item()) * bs
        correct += int(out.argmax(1).eq(y).sum().item())
        total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate(model, device, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            bs = int(y.size(0))
            loss_sum += float(loss.item()) * bs
            correct += int(out.argmax(1).eq(y).sum().item())
            total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--opt", choices=["sgd", "adam", "rmsprop"], default="sgd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)

    # IMPORTANT: CIFAR-100 preprocess dir
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/cifar100_resnet50_preprocess")
    ap.add_argument("--out_root", type=str, default="runs")

    # baseline optimizer hyperparams
    ap.add_argument("--lr", type=float, default=None, help="override default lr for chosen optimizer")
    ap.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (if opt=sgd)")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="optimizer weight_decay (baseline)")

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-cifar100")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default="cifar100_resnet50_baseline")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split, norm = load_artifacts(Path(args.preprocess_dir), args.data_seed)
    mean, std = norm["mean"], norm["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # CIFAR-100 via torchvision
    train_full = torchvision.datasets.CIFAR100(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        "data", train=False, download=True, transform=transform
    )

    train_dataset = Subset(train_full, split["train_idx"])
    val_dataset = Subset(train_full, split["val_idx"])

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
        val_shuffle=False,      # baseline: epoch-level eval
        train_drop_last=True,
        val_drop_last=False,
    )

    val_eval_loader = make_eval_loader(
        val_dataset,
        batch_size=args.bs,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = make_eval_loader(
        test_dataset,
        batch_size=args.bs,
        num_workers=4,
        pin_memory=True,
    )

    model = ResNet50CIFAR(num_classes=100).to(device)

    # Baseline optimizer defaults
    if args.opt == "sgd":
        lr = 0.1 if args.lr is None else float(args.lr)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    elif args.opt == "adam":
        lr = 1e-3 if args.lr is None else float(args.lr)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
    else:
        lr = 1e-3 if args.lr is None else float(args.lr)
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=float(args.weight_decay),
        )

    criterion = nn.CrossEntropyLoss()

    run_dir = (
        Path(args.out_root)
        / f"cifar100_resnet50_baseline_{args.opt}_seed{args.seed}_dataseed{args.data_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- logs ---
    time_logger = TimeLogger(run_dir / "time_log.csv")
    batch_logger = BatchLossLogger(run_dir / "curve.csv")

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n")

    # --- wandb ---
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --wandb was passed.")
        run_name = args.wandb_run_name or f"CIFAR100_ResNet50_baseline_{args.opt}_seed{args.seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config={
                "dataset": "CIFAR-100",
                "backbone": "ResNet50(CIFAR-stem, torchvision)",
                "method": "baseline",
                "optimizer": args.opt,
                "lr": lr,
                "momentum": float(args.momentum) if args.opt == "sgd" else None,
                "weight_decay": float(args.weight_decay),
                "seed": args.seed,
                "data_seed": args.data_seed,
                "batch_size": args.bs,
                "epochs": args.epochs,
                "preprocess_dir": args.preprocess_dir,
            },
        )

    time_logger.start()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer, batch_logger, epoch
        )
        val_loss, val_acc = evaluate(model, device, val_eval_loader, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        # time-based checklist log
        time_logger.log_epoch(
            train_loss, train_acc,
            val_loss, val_acc,
            test_loss, test_acc,
        )

        # epoch-based checklist log
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{train_loss:.8f},{train_acc:.6f},"
                f"{val_loss:.8f},{val_acc:.6f},"
                f"{test_loss:.8f},{test_acc:.6f}\n"
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
                },
                step=epoch,
            )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    batch_logger.close()

    final_val_loss, final_val_acc = evaluate(model, device, val_eval_loader, criterion)
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

    elapsed_total = float(time.time() - float(time_logger.start_time))

    result = {
        "dataset": "CIFAR-100",
        "backbone": "ResNet50(CIFAR-stem, torchvision)",
        "method": f"baseline_{args.opt}",
        "optimizer": args.opt,
        "lr": float(lr),
        "momentum": float(args.momentum) if args.opt == "sgd" else None,
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": float(final_val_acc),
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "elapsed_sec": elapsed_total,
        "run_dir": str(run_dir),
        "preprocess": str(Path(args.preprocess_dir) / f"seed_{args.data_seed}"),
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["final_val_acc"] = float(final_val_acc)
        wandb_run.summary["final_test_acc"] = float(final_test_acc)
        wandb_run.summary["elapsed_sec"] = float(elapsed_total)

        for p in [
            run_dir / "time_log.csv",
            run_dir / "train_log.csv",
            run_dir / "curve.csv",
            run_dir / "result.json",
        ]:
            if Path(p).exists():
                wandb.save(str(p))

        wandb_run.finish()


if __name__ == "__main__":
    main()

# utils/plot_exporter.py
"""
Plot Exporter
=============

Convert raw training logs into paper-level plotting interfaces.

INPUT (must exist in run_dir):
  - train_log.csv
  - time_log.csv
  - result.json (optional)

OUTPUT (written to run_dir):
  - metrics_epoch.csv
  - metrics_time.csv
  - meta.json

This module is MODEL-AGNOSTIC and METHOD-AGNOSTIC.
"""

from pathlib import Path
import csv
import json


# ---------------------- Core API ----------------------
def export_plot_files(
    run_dir: Path,
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    data_seed: int,
    epochs: int,
    optimizer: str = None,
):
    run_dir = Path(run_dir)

    _export_metrics_epoch(run_dir)
    _export_metrics_time(run_dir)
    _export_meta(
        run_dir,
        dataset=dataset,
        model=model,
        method=method,
        seed=seed,
        data_seed=data_seed,
        epochs=epochs,
        optimizer=optimizer,
    )


# ---------------------- Helpers ----------------------
def _export_metrics_epoch(run_dir: Path):
    src = run_dir / "train_log.csv"
    dst = run_dir / "metrics_epoch.csv"

    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    rows = []
    with open(src, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]),
                "test_loss": float(r["test_loss"]),
                "train_acc": float(r["train_acc"]),
                "test_acc": float(r["test_acc"]),
            })

    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_loss",
                "train_acc",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _export_metrics_time(run_dir: Path):
    src = run_dir / "time_log.csv"
    dst = run_dir / "metrics_time.csv"

    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    rows = []
    with open(src, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "seconds": float(r["elapsed_sec"]),
                "train_loss": float(r["train_loss"]),
                "test_loss": float(r["test_loss"]),
                "train_acc": float(r["train_acc"]),
                "test_acc": float(r["test_acc"]),
            })

    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seconds",
                "train_loss",
                "test_loss",
                "train_acc",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _export_meta(
    run_dir: Path,
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    data_seed: int,
    epochs: int,
    optimizer: str = None,
):
    meta = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": seed,
        "data_seed": data_seed,
        "epochs": epochs,
    }
    if optimizer is not None:
        meta["optimizer"] = optimizer

    dst = run_dir / "meta.json"
    with open(dst, "w") as f:
        json.dump(meta, f, indent=2)

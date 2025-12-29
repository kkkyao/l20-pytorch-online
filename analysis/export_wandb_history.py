import wandb
import pandas as pd
from pathlib import Path

ENTITY  = "leyao-li-epfl"
PROJECT = "l2o-online(new1)"

OUT_DIR = Path("wandb_history")
OUT_DIR.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

print("Start exporting runs (scan_history)...")

for run in runs:
    cfg = run.config or {}

    dataset  = cfg.get("dataset", "unknown")
    backbone = cfg.get("backbone", "unknown")
    method   = cfg.get("method", "unknown")
    seed     = cfg.get("seed", "unknown")
    alpha    = cfg.get("alpha", None)

    tag = f"{dataset}_{backbone}_{method}"
    if alpha is not None:
        tag += f"_alpha{alpha}"
    tag += f"_seed{seed}"
    tag = tag.replace("/", "_")

    print(f"  -> exporting {tag}")

    rows = []
    for row in run.scan_history():
        rows.append(row)

    if len(rows) == 0:
        print("     [skip] empty scan_history")
        continue

    df = pd.DataFrame(rows)

    # 只保留你关心的列（存在才保留）
    keep_cols = [
        "epoch",
        "_step",
        "_runtime",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df.to_csv(OUT_DIR / f"{tag}.csv", index=False)

print("Done.")

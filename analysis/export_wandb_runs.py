import wandb
import pandas as pd
from pathlib import Path

# ========= 配置 =========
ENTITY = "leyao-li-epfl-org"
PROJECT = "l2o-online(new1)"

OUT_DIR = Path("wandb_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 拉 runs =========
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

print(f"Found {len(runs)} runs")

# ========= 导出 =========
for run in runs:
    cfg = run.config or {}

    dataset  = cfg.get("dataset", "unknown")
    backbone = cfg.get("backbone", "unknown")
    method   = cfg.get("method", "unknown")
    seed     = cfg.get("seed", "unknown")

    tag = f"{dataset}_{backbone}_{method}_seed{seed}"
    tag = tag.replace("/", "_")

    try:
        history = run.history(keys=[
            "epoch",
            "_runtime",
            "train_loss",
            "test_loss",
            "train_acc",
            "test_acc",
        ])
    except Exception as e:
        print(f"[SKIP] {run.name}: {e}")
        continue

    out_path = OUT_DIR / f"{tag}.csv"
    history.to_csv(out_path, index=False)

    print(f"[OK] {out_path}")

import wandb
import pandas as pd
from pathlib import Path

# ===============================
# 已确认可用的 entity / project
# ===============================
ENTITY  = "leyao-li-epfl"
PROJECT = "l2o-online(new1)"

# ===============================
# 导出目录
# ===============================
OUT_DIR = Path("wandb_history")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 初始化 API
# ===============================
api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

print("Start exporting runs...")

for run in runs:
    cfg = run.config or {}

    dataset  = cfg.get("dataset", "unknown")
    backbone = cfg.get("backbone", "unknown")
    method   = cfg.get("method", "unknown")
    seed     = cfg.get("seed", "unknown")
    alpha    = cfg.get("alpha", None)

    # ---------- 文件名规范 ----------
    tag = f"{dataset}_{backbone}_{method}"
    if alpha is not None:
        tag += f"_alpha{alpha}"
    tag += f"_seed{seed}"

    # 防止路径非法
    tag = tag.replace("/", "_")

    print(f"  -> exporting {tag}")

    # ---------- 拉 history ----------
    history = run.history(keys=[
        "epoch",
        "_runtime",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
    ])

    # 有些 run 可能 history 为空（直接跳过）
    if history is None or len(history) == 0:
        print(f"     [skip] empty history")
        continue

    out_path = OUT_DIR / f"{tag}.csv"
    history.to_csv(out_path, index=False)

print("Done.")

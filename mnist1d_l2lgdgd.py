#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learning to Learn by Gradient Descent by Gradient Descent (L2L-GDGD)

- Coordinate-wise learned optimizer (LSTM)
- Optimizee: 1-hidden-layer MLP with sigmoid (paper-aligned)
- Data: MNIST-1D (length = 40)
- Meta-loss: sum of optimizee losses along optimization trajectory
- Truncated BPTT with unroll
- Added: argparse + wandb + run_dir logging + checkpoint saving
"""

import argparse
import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ---------------------- device / seed utils ----------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def w(x: torch.Tensor) -> torch.Tensor:
    """Move tensor to current device."""
    return x.to(get_device())


def detach_var(v: torch.Tensor) -> torch.Tensor:
    """
    Detach variable from previous graph but keep requires_grad.
    CRITICAL for truncated BPTT in L2L-GDGD.
    """
    out = Variable(v.detach().data, requires_grad=True)
    out.retain_grad()
    return out.to(get_device())


# ---------------------- MNIST-1D task ----------------------
class MNIST1DLoss:
    """
    Task = data stream.
    Each task corresponds to one training run with its own minibatch sequence.
    """

    def __init__(self, training: bool = True, batch_size: int = 128, data_seed: int = 42):
        (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=data_seed)
        if training:
            x, y = xtr, ytr
        else:
            x, y = xte, yte

        # Always float32 / int64
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))

        self.loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._iter = iter(self.loader)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ---------------------- Optimizee (paper-aligned MLP) ----------------------
class MNIST1DOptimizee(nn.Module):
    """
    Functional MLP optimizee:
        input(40) -> hidden(20, sigmoid) -> output(10)

    IMPORTANT:
      - Parameters stored in a plain dict are NOT auto-moved by .to(device).
      - So when params is None, we MUST create them on the correct device.
    """

    def __init__(self, hidden_dim: int = 20, params: Optional[Dict[str, nn.Parameter]] = None):
        super().__init__()
        device = get_device()
        if params is None:
            self.params = {
                "W1": nn.Parameter(torch.randn(40, hidden_dim, device=device) * 0.01),
                "b1": nn.Parameter(torch.zeros(hidden_dim, device=device)),
                "W2": nn.Parameter(torch.randn(hidden_dim, 10, device=device) * 0.01),
                "b2": nn.Parameter(torch.zeros(10, device=device)),
            }
        else:
            self.params = params

    def all_named_parameters(self):
        return list(self.params.items())

    def forward(self, loss_obj: MNIST1DLoss) -> torch.Tensor:
        x, y = loss_obj.sample()
        x = x.to(get_device())
        y = y.to(get_device())

        h = torch.sigmoid(x @ self.params["W1"] + self.params["b1"])
        logits = h @ self.params["W2"] + self.params["b2"]
        return F.cross_entropy(logits, y)


# ---------------------- Learned Optimizer ----------------------
class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise LSTM optimizer (paper-aligned).
    """

    def __init__(self, hidden_sz: int = 20, preproc: bool = True, preproc_factor: float = 10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = math.exp(-preproc_factor)

        if preproc:
            self.lstm1 = nn.LSTMCell(2, hidden_sz)
        else:
            self.lstm1 = nn.LSTMCell(1, hidden_sz)

        self.lstm2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.out = nn.Linear(hidden_sz, 1)

    def _preprocess(self, g: torch.Tensor) -> torch.Tensor:
        g = g.data
        out = torch.zeros(g.size(0), 2, device=g.device)
        keep = (g.abs() >= self.preproc_threshold).squeeze()

        out[:, 0][keep] = (torch.log(g.abs()[keep] + 1e-8) / self.preproc_factor).squeeze()
        out[:, 1][keep] = torch.sign(g[keep]).squeeze()

        out[:, 0][~keep] = -1
        out[:, 1][~keep] = (math.exp(self.preproc_factor) * g[~keep]).squeeze()

        return Variable(out)

    def forward(
        self,
        grad: torch.Tensor,
        hidden: List[torch.Tensor],
        cell: List[torch.Tensor],
    ):
        if self.preproc:
            grad = self._preprocess(grad)

        h0, c0 = self.lstm1(grad, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)
        return update, (h0, h1), (c0, c1)


# ---------------------- Meta-training (single trajectory) ----------------------
@torch.no_grad()
def _count_params(optimizee: MNIST1DOptimizee) -> int:
    return sum(int(np.prod(p.size())) for _, p in optimizee.all_named_parameters())


def do_fit(
    opt_net: LearnedOptimizer,
    meta_opt: Optional[optim.Optimizer],
    optim_steps: int = 100,
    unroll: int = 20,
    out_mul: float = 0.1,
    training: bool = True,
    batch_size: int = 128,
    data_seed: int = 42,
) -> List[float]:
    """
    Run one optimizee trajectory for optim_steps steps.
    Return list of per-step optimizee losses (python floats).
    """
    device = get_device()
    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1  # paper-style eval: no unroll training

    loss_obj = MNIST1DLoss(training=training, batch_size=batch_size, data_seed=data_seed)
    optimizee = MNIST1DOptimizee().to(device)  # module itself has no parameters tracked, but OK

    n_params = _count_params(optimizee)
    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    if training and meta_opt is not None:
        meta_opt.zero_grad(set_to_none=True)

    total_loss = None
    losses_ever: List[float] = []

    for step in range(1, optim_steps + 1):
        loss = optimizee(loss_obj)
        losses_ever.append(float(loss.item()))

        total_loss = loss if total_loss is None else (total_loss + loss)
        loss.backward(retain_graph=training)

        offset = 0
        new_params: Dict[str, nn.Parameter] = {}
        new_hidden = [torch.zeros_like(h) for h in hidden]
        new_cell = [torch.zeros_like(c) for c in cell]

        for name, p in optimizee.all_named_parameters():
            sz = int(np.prod(p.size()))
            if p.grad is None:
                raise RuntimeError(f"Gradient is None for parameter {name}")

            grad = detach_var(p.grad.view(sz, 1))

            update, h_new, c_new = opt_net(
                grad,
                [h[offset:offset + sz] for h in hidden],
                [c[offset:offset + sz] for c in cell],
            )

            for i in range(2):
                new_hidden[i][offset:offset + sz] = h_new[i]
                new_cell[i][offset:offset + sz] = c_new[i]

            new_p = p + out_mul * update.view_as(p)
            new_p.retain_grad()
            new_params[name] = new_p

            offset += sz

        if step % unroll == 0:
            if training and meta_opt is not None:
                meta_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                meta_opt.step()

            optimizee = MNIST1DOptimizee(params={k: detach_var(v) for k, v in new_params.items()})
            hidden = [detach_var(h) for h in new_hidden]
            cell = [detach_var(c) for c in new_cell]
            total_loss = None
        else:
            optimizee = MNIST1DOptimizee(params=new_params)
            hidden = new_hidden
            cell = new_cell

    return losses_ever


# ---------------------- Meta-training entry ----------------------
def train_l2lgdgd(
    seed: int = 0,
    data_seed: int = 42,
    meta_epochs: int = 50,
    inner_loops_per_epoch: int = 20,
    eval_tasks: int = 10,
    optim_steps: int = 100,
    unroll: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    out_mul: float = 0.1,
    save_dir: Optional[Path] = None,
    use_wandb: bool = False,
) -> Tuple[float, Dict]:
    """
    Returns:
      best_loss, best_state_dict
    """
    set_seed(seed)
    device = get_device()

    opt_net = LearnedOptimizer().to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_loss = float("inf")
    best_state: Optional[Dict] = None

    for ep in range(meta_epochs):
        # meta-train
        for _ in range(inner_loops_per_epoch):
            do_fit(
                opt_net,
                meta_opt,
                optim_steps=optim_steps,
                unroll=unroll,
                out_mul=out_mul,
                training=True,
                batch_size=batch_size,
                data_seed=data_seed,
            )

        # meta-eval (held-out tasks) – here: re-sampling batches, same distribution
        eval_losses = []
        for _ in range(eval_tasks):
            traj = do_fit(
                opt_net,
                meta_opt=None,
                optim_steps=optim_steps,
                unroll=unroll,
                out_mul=out_mul,
                training=False,
                batch_size=batch_size,
                data_seed=data_seed,
            )
            eval_losses.append(sum(traj))
        eval_loss = float(np.mean(eval_losses))

        print(f"[MetaEpoch {ep:03d}] eval meta-loss = {eval_loss:.6f}")

        if use_wandb and wandb is not None:
            wandb.log({"meta_epoch": ep, "eval_meta_loss": eval_loss})

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = copy.deepcopy(opt_net.state_dict())

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / "best_opt.pt"
                torch.save(best_state, ckpt_path)

    if best_state is None:
        best_state = opt_net.state_dict()

    return best_loss, best_state


# ---------------------- CLI / main ----------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--meta_epochs", type=int, default=50)
    ap.add_argument("--inner_loops_per_epoch", type=int, default=20)
    ap.add_argument("--eval_tasks", type=int, default=10)
    ap.add_argument("--optim_steps", type=int, default=100)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=128, help="batch size")
    ap.add_argument("--out_mul", type=float, default=0.1)
    ap.add_argument("--runs_dir", type=str, default="runs", help="base directory for outputs")

    # wandb
    ap.add_argument("--wandb", action="store_true", help="enable wandb logging")
    ap.add_argument("--wandb_project", type=str, default="l2o-online")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2lgdgd")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    return ap


def main():
    args = build_argparser().parse_args()

    device = get_device()
    print(f"[INFO] device = {device}")

    run_name = (
        f"mnist1d_l2lgdgd_seed{args.seed}_data{args.data_seed}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    # wandb init
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config={
                "seed": args.seed,
                "data_seed": args.data_seed,
                "meta_epochs": args.meta_epochs,
                "inner_loops_per_epoch": args.inner_loops_per_epoch,
                "eval_tasks": args.eval_tasks,
                "optim_steps": args.optim_steps,
                "unroll": args.unroll,
                "lr": args.lr,
                "bs": args.bs,
                "out_mul": args.out_mul,
            },
        )

    start = time.time()
    best_loss, best_state = train_l2lgdgd(
        seed=args.seed,
        data_seed=args.data_seed,
        meta_epochs=args.meta_epochs,
        inner_loops_per_epoch=args.inner_loops_per_epoch,
        eval_tasks=args.eval_tasks,
        optim_steps=args.optim_steps,
        unroll=args.unroll,
        lr=args.lr,
        batch_size=args.bs,
        out_mul=args.out_mul,
        save_dir=run_dir,
        use_wandb=bool(args.wandb and wandb is not None),
    )
    elapsed = time.time() - start

    print(f"[RESULT] Best meta-loss: {best_loss:.6f}  (elapsed={elapsed/60:.2f} min)")

    # save result json
    result = {
        "method": "mnist1d_l2lgdgd",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "meta_epochs": int(args.meta_epochs),
        "inner_loops_per_epoch": int(args.inner_loops_per_epoch),
        "eval_tasks": int(args.eval_tasks),
        "optim_steps": int(args.optim_steps),
        "unroll": int(args.unroll),
        "lr": float(args.lr),
        "bs": int(args.bs),
        "out_mul": float(args.out_mul),
        "best_meta_loss": float(best_loss),
        "elapsed_sec": float(elapsed),
        "run_dir": str(run_dir),
        "device": str(get_device()),
        "checkpoint": str(run_dir / "best_opt.pt"),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_meta_loss"] = float(best_loss)
        wandb_run.summary["elapsed_sec"] = float(elapsed)
        # upload key files
        for p in [run_dir / "result.json", run_dir / "best_opt.pt"]:
            if p.exists():
                wandb.save(str(p))
        wandb_run.finish()


if __name__ == "__main__":
    main()

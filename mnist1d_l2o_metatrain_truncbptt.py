#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D meta-training of a learned optimizer (coordinate-wise 2-layer LSTM)
with long trajectory T and truncated BPTT segments of length K.

Paper alignment (Andrychowicz et al., 2016):
  - Optimizer is coordinate-wise 2-layer LSTM (shared weights across coordinates).
  - Meta objective is trajectory loss: sum_{t=1..T} w_t f_t(theta_t).
  - Truncated BPTT: backprop through K unrolled steps, update phi, then detach and continue.

This implementation:
  - Episode: sample a task_seed, init a fresh optimizee, reset optimizer hidden states,
            run T inner steps.
  - Every K steps: meta_loss_segment.backward(); meta_opt.step(); detach (params + hidden states),
                   continue the remaining steps of the SAME optimizee (long trajectory).

Metrics printing:
  - We define "meta_epoch" as running episodes_per_epoch episodes.
  - Each meta_epoch prints train_loss/acc (averaged over all inner steps of meta-training in that epoch),
    and test_loss/acc by evaluating learned optimizer on a held-out eval task.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# ---------------- functional_call compatibility ----------------
def _get_functional_call():
    try:
        from torch.func import functional_call as fc  # type: ignore
        return fc
    except Exception:
        from torch.nn.utils.stateless import functional_call as fc  # type: ignore
        return fc


functional_call = _get_functional_call()


# ------------------------------- Utilities -------------------------------

def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def to_NL(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == length:
        return x
    if x.ndim == 3:
        if x.shape[1] == length and x.shape[2] == 1:
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == length:
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}")


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mean[None, :]) / (std[None, :] + eps)


def ensure_split_and_norm(
    preprocess_dir: Path,
    task_seed: int,
    xtr: np.ndarray,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure split.json and norm.json exist for this task_seed.
    If not, create:
      - train_idx/val_idx from xtr
      - mean/std computed on train split only
    Returns: train_idx, val_idx, mean, std
    """
    pdir = Path(preprocess_dir) / f"seed_{task_seed}"
    pdir.mkdir(parents=True, exist_ok=True)
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"

    n = xtr.shape[0]
    if split_path.exists() and norm_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        with open(norm_path, "r", encoding="utf-8") as f:
            norm = json.load(f)
        train_idx = np.array(split["train_idx"], dtype=np.int64)
        val_idx = np.array(split["val_idx"], dtype=np.int64)
        mean = np.array(norm["mean"], dtype=np.float32)
        std = np.array(norm["std"], dtype=np.float32)
        return train_idx, val_idx, mean, std

    rng = np.random.RandomState(task_seed)
    perm = rng.permutation(n)
    v = int(round(val_frac * n))
    v = max(1, min(v, n - 1))
    val_idx = perm[:v]
    train_idx = perm[v:]

    x_train = xtr[train_idx]
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f, indent=2)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    return train_idx, val_idx, mean, std


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for b in loader:
            yield b


# --------------------------- MLP backbone ---------------------------

def build_mlp(input_len: int = 40, num_layers: int = 5, width: int = 128) -> nn.Module:
    act_layer = nn.ReLU

    class MLPBackbone(nn.Module):
        def __init__(self, length: int):
            super().__init__()
            layers = []
            in_dim = length
            for _ in range(max(0, num_layers)):
                layers.append(nn.Linear(in_dim, width))
                layers.append(act_layer(inplace=True))
                in_dim = width
            self.mlp = nn.Sequential(*layers)
            self.out = nn.Linear(in_dim, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.mlp(x)
            return self.out(x)

    return MLPBackbone(input_len)


# ---------------------- Learned Optimizer (coordinate-wise LSTM) ----------------------

class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise 2-layer LSTM optimizer with optional gradient preprocessing.

    detach_grad_input=True corresponds to "dropping dashed edges" (first-order approx).
    """
    def __init__(
        self,
        hidden_sz: int = 20,
        preproc: bool = True,
        preproc_factor: float = 10.0,
        detach_grad_input: bool = True,
    ):
        super().__init__()
        self.hidden_sz = int(hidden_sz)
        self.preproc = bool(preproc)
        self.preproc_factor = float(preproc_factor)
        self.preproc_threshold = float(np.exp(-self.preproc_factor))
        self.detach_grad_input = bool(detach_grad_input)

        in_dim = 2 if self.preproc else 1
        self.lstm1 = nn.LSTMCell(in_dim, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz, 1)
        nn.init.constant_(self.out.bias, 0.0)

    def _maybe_detach(self, g: torch.Tensor) -> torch.Tensor:
        return g.detach() if self.detach_grad_input else g

    def _preprocess(self, g: torch.Tensor) -> torch.Tensor:
        gg = self._maybe_detach(g)
        out = torch.zeros(gg.size(0), 2, device=gg.device, dtype=gg.dtype)
        keep = (gg.abs() >= self.preproc_threshold).view(-1)

        out[keep, 0] = (torch.log(gg[keep].abs() + 1e-8) / self.preproc_factor).view(-1)
        out[keep, 1] = torch.sign(gg[keep]).view(-1)

        out[~keep, 0] = -1.0
        out[~keep, 1] = (float(np.exp(self.preproc_factor)) * gg[~keep]).view(-1)
        return out

    def forward(self, grad_vec: torch.Tensor, hidden, cell):
        if self.preproc:
            x = self._preprocess(grad_vec)   # [N,2]
        else:
            x = self._maybe_detach(grad_vec) # [N,1]
        h0, c0 = self.lstm1(x, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)  # [N,1]
        return update, (h0, h1), (c0, c1)


def named_params_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in module.named_parameters()}


def grads_to_vector(grads: List[torch.Tensor], params: List[torch.Tensor]) -> torch.Tensor:
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs, dim=0).view(-1, 1)  # [N,1]


def vector_to_params(vec: torch.Tensor, params_template: List[torch.Tensor]) -> List[torch.Tensor]:
    outs = []
    offset = 0
    v = vec.view(-1)
    for p in params_template:
        sz = p.numel()
        outs.append(v[offset:offset + sz].view_as(p))
        offset += sz
    return outs


def clip_delta_vec(delta_vec: torch.Tensor, clip_update: float, eps: float = 1e-12):
    norm = torch.linalg.vector_norm(delta_vec.view(-1)).clamp_min(eps)
    if clip_update <= 0.0:
        coef = torch.ones((), device=delta_vec.device, dtype=delta_vec.dtype)
        return delta_vec, norm, coef
    clip_t = torch.tensor(float(clip_update), device=delta_vec.device, dtype=delta_vec.dtype)
    coef = torch.clamp(clip_t / norm, max=1.0)
    return delta_vec * coef, norm, coef


# ------------------------------ Evaluation ------------------------------

def eval_learned_opt(
    *,
    length: int,
    layers: int,
    width: int,
    init_state_dict: Dict[str, torch.Tensor],
    opt_net: LearnedOptimizer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader,
    device: torch.device,
    out_mul: float,
    clip_update: float,
) -> Tuple[float, float]:
    """
    Run learned optimizer for len(train_batches) steps, then evaluate on test.
    Need gradients inside inner loop; only final test uses no_grad.
    """
    ce = nn.CrossEntropyLoss()

    m = build_mlp(input_len=length, num_layers=layers, width=width).to(device)
    m.load_state_dict(init_state_dict, strict=True)

    params_dict = {k: v.detach().to(device).requires_grad_(True) for k, v in named_params_dict(m).items()}
    params_list_template = [params_dict[k] for k in params_dict.keys()]
    n_params = int(sum(p.numel() for p in params_list_template))

    hidden = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]

    opt_net.eval()

    for xb, yb in train_batches:
        xb = xb.to(device); yb = yb.to(device)

        logits = functional_call(m, params_dict, (xb,))
        loss = ce(logits, yb)

        params_list = [params_dict[k] for k in params_dict.keys()]
        grads = torch.autograd.grad(loss, params_list, create_graph=False, retain_graph=False, allow_unused=False)
        g_vec = grads_to_vector(list(grads), params_list)

        upd_vec, h_new, c_new = opt_net(g_vec, hidden, cell)
        delta_vec = upd_vec * float(out_mul)
        delta_vec, _, _ = clip_delta_vec(delta_vec, float(clip_update))

        delta_list = vector_to_params(delta_vec, params_list)
        params_dict = {name: p + dp for (name, p), dp in zip(params_dict.items(), delta_list)}

        hidden = [h.detach() for h in h_new]
        cell = [c.detach() for c in c_new]
        params_dict = {k: v.detach().requires_grad_(True) for k, v in params_dict.items()}

    # test eval
    m.eval()
    ce = nn.CrossEntropyLoss()
    correct, total, loss_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = functional_call(m, params_dict, (xb,))
            loss = ce(logits, yb)
            loss_sum += float(loss.item()); n += 1
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return loss_sum / max(n, 1), correct / max(total, 1)


def eval_baseline_sgd(
    *,
    length: int,
    layers: int,
    width: int,
    init_state_dict: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
) -> Tuple[float, float]:
    """
    SGD baseline: run 'steps' minibatch updates, then test.
    """
    m = build_mlp(input_len=length, num_layers=layers, width=width).to(device)
    m.load_state_dict(init_state_dict, strict=True)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    m.train()
    it = infinite_loader(train_loader)
    for _ in range(int(steps)):
        xb, yb = next(it)
        xb = xb.to(device); yb = yb.to(device)
        logits = m(xb)
        loss = ce(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    m.eval()
    correct, total, loss_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = m(xb)
            loss = ce(logits, yb)
            loss_sum += float(loss.item()); n += 1
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return loss_sum / max(n, 1), correct / max(total, 1)


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()

    # global
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")
    ap.add_argument("--val_frac", type=float, default=0.2)

    # task distribution
    ap.add_argument("--task_seed_min", type=int, default=0)
    ap.add_argument("--task_seed_max", type=int, default=99)

    # meta schedule
    ap.add_argument("--meta_epochs", type=int, default=20)
    ap.add_argument("--episodes_per_epoch", type=int, default=50)

    # long trajectory + truncation
    ap.add_argument("--T", type=int, default=200, help="Long trajectory length per episode.")
    ap.add_argument("--K", type=int, default=20, help="Truncation length for BPTT (segment length).")
    ap.add_argument("--wt", type=float, default=1.0)

    # optimizee
    ap.add_argument("--mlp_layers", type=int, default=5)
    ap.add_argument("--mlp_width", type=int, default=128)
    ap.add_argument("--init_normal_std", type=float, default=0.0, help="If >0, init optimizee params ~ N(0, std).")

    # learned optimizer
    ap.add_argument("--opt_hidden_sz", type=int, default=20)
    ap.add_argument("--opt_lr", type=float, default=1e-3)
    ap.add_argument("--out_mul", type=float, default=1e-2)
    ap.add_argument("--clip_update", type=float, default=0.1)
    ap.add_argument("--clip_phi_grad", type=float, default=1.0)
    ap.add_argument("--no_preproc", action="store_true")
    ap.add_argument("--preproc_factor", type=float, default=10.0)
    ap.add_argument("--detach_grad_input", action="store_true")

    # evaluation each meta_epoch
    ap.add_argument("--eval_task_seed", type=int, default=123)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--sgd_lr", type=float, default=1e-2)

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-mnist1d")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2o_longtraj_truncbptt")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    set_seed(args.seed)

    run_name = (
        f"mnist1d_l2o_longtraj_seed{args.seed}_task{args.task_seed_min}-{args.task_seed_max}_"
        f"T{args.T}_K{args.K}_lr{args.opt_lr}_out{args.out_mul}_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir = {run_dir}")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed, but --wandb was passed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    opt_net = LearnedOptimizer(
        hidden_sz=args.opt_hidden_sz,
        preproc=(not args.no_preproc),
        preproc_factor=args.preproc_factor,
        detach_grad_input=bool(args.detach_grad_input),
    ).to(device)
    meta_opt = torch.optim.Adam(opt_net.parameters(), lr=args.opt_lr)
    ce = nn.CrossEntropyLoss()

    # logs
    train_log = run_dir / "meta_epoch_log.csv"
    with open(train_log, "w", encoding="utf-8") as f:
        f.write("meta_epoch,train_loss,train_acc,learned_test_loss,learned_test_acc,sgd_test_loss,sgd_test_acc\n")

    rng = np.random.RandomState(args.seed)
    start_time = time.time()

    # pre-build eval task loaders each epoch (same task_seed)
    def build_task_loaders(task_seed: int):
        (xtr, ytr), (xte, yte) = load_mnist1d(length=args.length, seed=task_seed)
        xtr = to_NL(xtr, args.length).astype(np.float32, copy=False)
        xte = to_NL(xte, args.length).astype(np.float32, copy=False)
        ytr = np.asarray(ytr, dtype=np.int64)
        yte = np.asarray(yte, dtype=np.int64)

        train_idx, _, mean, std = ensure_split_and_norm(
            preprocess_dir=Path(args.preprocess_dir),
            task_seed=task_seed,
            xtr=xtr,
            val_frac=float(args.val_frac),
        )

        x_train = apply_norm_np(xtr[train_idx], mean, std)
        y_train = ytr[train_idx]
        x_test = apply_norm_np(xte, mean, std)
        y_test = yte

        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        x_test_t = torch.from_numpy(x_test)
        y_test_t = torch.from_numpy(y_test)

        gen = torch.Generator()
        gen.manual_seed(int(args.seed * 999 + task_seed))

        train_loader = DataLoader(
            TensorDataset(x_train_t, y_train_t),
            batch_size=128,
            shuffle=True,
            drop_last=True,
            generator=gen,
        )
        test_loader = DataLoader(
            TensorDataset(x_test_t, y_test_t),
            batch_size=512,
            shuffle=False,
        )
        return train_loader, test_loader

    eval_train_loader, eval_test_loader = build_task_loaders(int(args.eval_task_seed))

    # fixed init for eval comparisons
    torch.manual_seed(int(args.eval_task_seed))
    init_model = build_mlp(input_len=args.length, num_layers=args.mlp_layers, width=args.mlp_width)
    init_state_dict = {k: v.detach().cpu() for k, v in init_model.state_dict().items()}

    for meta_epoch in range(1, int(args.meta_epochs) + 1):
        opt_net.train()

        # aggregate training metrics over this meta_epoch
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_steps = 0

        for epi in range(int(args.episodes_per_epoch)):
            task_seed = int(rng.randint(args.task_seed_min, args.task_seed_max + 1))

            # ---- sample task ----
            (xtr, ytr), _ = load_mnist1d(length=args.length, seed=task_seed)
            xtr = to_NL(xtr, args.length).astype(np.float32, copy=False)
            ytr = np.asarray(ytr, dtype=np.int64)

            train_idx, _, mean, std = ensure_split_and_norm(
                preprocess_dir=Path(args.preprocess_dir),
                task_seed=task_seed,
                xtr=xtr,
                val_frac=float(args.val_frac),
            )

            x_train = apply_norm_np(xtr[train_idx], mean, std)
            y_train = ytr[train_idx]
            x_train_t = torch.from_numpy(x_train)
            y_train_t = torch.from_numpy(y_train)

            gen = torch.Generator()
            gen.manual_seed(int(args.seed * 10_000 + meta_epoch * 1_000 + epi))
            train_loader = DataLoader(
                TensorDataset(x_train_t, y_train_t),
                batch_size=128,
                shuffle=True,
                drop_last=True,
                generator=gen,
            )
            train_iter = infinite_loader(train_loader)

            # ---- fresh optimizee init ----
            net = build_mlp(input_len=args.length, num_layers=args.mlp_layers, width=args.mlp_width).to(device)
            if args.init_normal_std and float(args.init_normal_std) > 0.0:
                with torch.no_grad():
                    for p in net.parameters():
                        p.normal_(mean=0.0, std=float(args.init_normal_std))

            params_dict = {k: v.detach().to(device).requires_grad_(True) for k, v in named_params_dict(net).items()}
            params_list_template = [params_dict[k] for k in params_dict.keys()]
            n_params = int(sum(p.numel() for p in params_list_template))

            # reset optimizer hidden state each episode (typical meta-training setting)
            hidden = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]
            cell = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]

            # ---- long trajectory with truncated BPTT ----
            seg_meta_loss = torch.zeros((), device=device)
            last_coef = 1.0

            for t in range(1, int(args.T) + 1):
                xb, yb = next(train_iter)
                xb = xb.to(device); yb = yb.to(device)

                logits = functional_call(net, params_dict, (xb,))
                loss = ce(logits, yb)

                # accumulate trajectory objective (Eq.3)
                seg_meta_loss = seg_meta_loss + float(args.wt) * loss

                # metrics (detached)
                with torch.no_grad():
                    epoch_loss_sum += float(loss.item())
                    pred = logits.argmax(dim=1)
                    epoch_correct += int((pred == yb).sum().item())
                    epoch_total += int(yb.size(0))
                    epoch_steps += 1

                params_list = [params_dict[k] for k in params_dict.keys()]
                create_graph = (not bool(args.detach_grad_input))
                # retain_graph=True is important because loss is part of seg_meta_loss
                grads = torch.autograd.grad(
                    loss,
                    params_list,
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=False,
                )
                g_vec = grads_to_vector(list(grads), params_list)

                upd_vec, h_new, c_new = opt_net(g_vec, hidden, cell)
                delta_vec = upd_vec * float(args.out_mul)
                delta_vec, _, coef = clip_delta_vec(delta_vec, float(args.clip_update))
                last_coef = float(coef.detach().item())

                delta_list = vector_to_params(delta_vec, params_list)
                params_dict = {name: p + dp for (name, p), dp in zip(params_dict.items(), delta_list)}

                hidden = list(h_new)
                cell = list(c_new)

                # ---- every K steps: outer update + detach (truncate) ----
                if (t % int(args.K)) == 0:
                    meta_opt.zero_grad(set_to_none=True)
                    seg_meta_loss.backward()
                    if args.clip_phi_grad and float(args.clip_phi_grad) > 0.0:
                        torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(args.clip_phi_grad))
                    meta_opt.step()

                    # truncate graph: detach optimizee params and optimizer states
                    params_dict = {k: v.detach().requires_grad_(True) for k, v in params_dict.items()}
                    hidden = [h.detach() for h in hidden]
                    cell = [c.detach() for c in cell]

                    # reset segment meta loss
                    seg_meta_loss = torch.zeros((), device=device)

            # If T not divisible by K, do one last outer update on remaining segment
            if (int(args.T) % int(args.K)) != 0:
                meta_opt.zero_grad(set_to_none=True)
                seg_meta_loss.backward()
                if args.clip_phi_grad and float(args.clip_phi_grad) > 0.0:
                    torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(args.clip_phi_grad))
                meta_opt.step()

        # ---- meta_epoch summary train metrics ----
        train_loss_epoch = epoch_loss_sum / max(epoch_steps, 1)
        train_acc_epoch = epoch_correct / max(epoch_total, 1)

        # ---- evaluation each meta_epoch ----
        # fixed list of eval train batches for learned optimizer eval
        eval_it = infinite_loader(eval_train_loader)
        eval_batches = [next(eval_it) for _ in range(int(args.eval_steps))]

        learned_test_loss, learned_test_acc = eval_learned_opt(
            length=args.length,
            layers=args.mlp_layers,
            width=args.mlp_width,
            init_state_dict=init_state_dict,
            opt_net=opt_net,
            train_batches=eval_batches,
            test_loader=eval_test_loader,
            device=device,
            out_mul=float(args.out_mul),
            clip_update=float(args.clip_update),
        )

        sgd_test_loss, sgd_test_acc = eval_baseline_sgd(
            length=args.length,
            layers=args.mlp_layers,
            width=args.mlp_width,
            init_state_dict=init_state_dict,
            train_loader=eval_train_loader,
            test_loader=eval_test_loader,
            device=device,
            steps=int(args.eval_steps),
            lr=float(args.sgd_lr),
        )

        with open(train_log, "a", encoding="utf-8") as f:
            f.write(
                f"{meta_epoch},{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{learned_test_loss:.8f},{learned_test_acc:.6f},"
                f"{sgd_test_loss:.8f},{sgd_test_acc:.6f}\n"
            )

        elapsed = time.time() - start_time
        print(
            f"[META_EPOCH {meta_epoch:03d}] "
            f"train={train_loss_epoch:.4f}/{train_acc_epoch:.4f} "
            f"| learned_test={learned_test_loss:.4f}/{learned_test_acc:.4f} "
            f"| sgd_test={sgd_test_loss:.4f}/{sgd_test_acc:.4f} "
            f"(elapsed={elapsed/60:.2f} min)"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "meta_epoch": meta_epoch,
                    "train/loss": train_loss_epoch,
                    "train/acc": train_acc_epoch,
                    "eval/learned_test_loss": learned_test_loss,
                    "eval/learned_test_acc": learned_test_acc,
                    "eval/sgd_test_loss": sgd_test_loss,
                    "eval/sgd_test_acc": sgd_test_acc,
                    "time/elapsed_sec": elapsed,
                    "meta/T": int(args.T),
                    "meta/K": int(args.K),
                    "meta/episodes_per_epoch": int(args.episodes_per_epoch),
                }
            )

    if wandb_run is not None:
        wandb_run.finish()

    print(f"[DONE] logs in: {run_dir}")


if __name__ == "__main__":
    main()

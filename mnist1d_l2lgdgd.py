#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learned optimizer (coordinate-wise 2-layer LSTM) trained with truncated BPTT
through the optimizee trajectory (Route B, aligned to Andrychowicz et al. 2016 Eq.(3)).

Key points (paper alignment):
  - Meta objective is trajectory loss: sum_{t=1..T} w_t f(theta_t), Eq.(3). We use w_t=1 by default.
  - Optimizer is coordinate-wise 2-layer LSTM with shared weights across coordinates (Fig.3).
  - Truncated BPTT: unroll K steps, backprop meta-loss through the unrolled graph, then detach
    optimizee parameters and optimizer hidden states before next segment.
  - (Optional, default OFF) We drop gradients along dashed edges (first-order approximation):
      assume ∂(∇_t)/∂phi = 0 by detaching gradient input to the optimizer.
    This matches the paper's description of dropping dashed edges to avoid second derivatives.

Extra (engineering knobs):
  - Optional global-norm clipping of update Δθ (not required by the paper; default disabled).
  - Optional meta objective augmentation with val loss at the end of unroll (B2-style), default 0.
  - Optional reset of optimizer state each epoch / interval.

Reference: "Learning to learn by gradient descent by gradient descent", Andrychowicz et al., NIPS 2016.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

# -------- functional_call compatibility (torch 2.x) --------
try:
    # torch<=2.0 / 2.1 often has this
    from torch.nn.utils.stateless import functional_call as _functional_call  # type: ignore
    def functional_call(module: nn.Module, params: Dict[str, torch.Tensor], args: Tuple[torch.Tensor, ...]):
        return _functional_call(module, params, args)
except Exception:
    # torch.func.functional_call (torch>=2.0)
    try:
        from torch.func import functional_call as _functional_call  # type: ignore
        def functional_call(module: nn.Module, params: Dict[str, torch.Tensor], args: Tuple[torch.Tensor, ...]):
            return _functional_call(module, params, args)
    except Exception as e:
        raise ImportError("Could not import functional_call from torch. Please use torch>=2.0.") from e


# ------------------------------- Utilities -------------------------------

class BatchLossLogger:
    def __init__(self, run_dir: Path, meta: dict, filename: str, flush_every: int = 200):
        self.run_dir = Path(run_dir)
        self.meta = meta
        self.flush_every = flush_every
        self.global_iter = 0
        self.curr_epoch = 0
        self.rows = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.curve_path = self.run_dir / filename
        with open(self.curve_path, "w", encoding="utf-8") as f:
            f.write("iter,epoch,loss,acc,method,seed,opt,lr\n")

    def on_epoch_begin(self, epoch: int):
        self.curr_epoch = int(epoch)

    def on_train_step_end(self, loss_value: float, acc_value: float):
        row = [
            self.global_iter,
            self.curr_epoch,
            float(loss_value),
            float(acc_value),
            str(self.meta.get("method", "unknown")).lower(),
            int(self.meta.get("seed", -1)),
            str(self.meta.get("opt", "unknown")).lower(),
            float(self.meta.get("lr", float("nan"))),
        ]
        self.rows.append(row)
        self.global_iter += 1
        if len(self.rows) >= self.flush_every:
            self._flush()

    def on_train_end(self):
        if self.rows:
            self._flush()

    def _flush(self):
        with open(self.curve_path, "a", encoding="utf-8") as f:
            for r in self.rows:
                f.write(",".join(map(str, r)) + "\n")
        self.rows.clear()


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_artifacts(preprocess_dir: Path, seed: int):
    pdir = Path(preprocess_dir) / f"seed_{seed}"
    split_path = pdir / "split.json"
    norm_path = pdir / "norm.json"
    if not split_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Preprocess artifacts not found under: {pdir}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    return split, norm


def apply_norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mean[None, :]) / (std[None, :] + eps)


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


# --------------------------- MLP backbone ---------------------------

def build_mlp(input_len: int = 40, num_layers: int = 5, width: int = 128, activation: str = "relu") -> nn.Module:
    act_layer = nn.ReLU if activation.lower() == "relu" else nn.ReLU

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
    Coordinate-wise 2-layer LSTM optimizer (Fig.3 in the paper).

    Gradient preprocessing (Appendix A):
      g -> (log(|g|)/p, sign(g)) if |g| >= exp(-p)
           (-1, exp(p)*g)       otherwise

    NOTE on "drop dashed edges" (paper):
      You can detach the gradient input so that ∂(∇_t)/∂phi = 0 (no second derivatives).
      This is controlled by `detach_grad_input`.
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
            x = self._preprocess(grad_vec)  # [N,2]
        else:
            x = self._maybe_detach(grad_vec)  # [N,1]
        h0, c0 = self.lstm1(x, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)  # [N,1]
        return update, (h0, h1), (c0, c1)


def named_params_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in module.named_parameters()}


def params_to_vector(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params], dim=0).view(-1, 1)  # [N,1]


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


def clip_delta_vec(delta_vec: torch.Tensor, clip_update: Optional[float], eps: float = 1e-12):
    """
    Differentiable global-norm clipping for delta (Δθ).
    Returns:
      delta_clipped, norm_before, coef
    """
    norm = torch.linalg.vector_norm(delta_vec.view(-1)).clamp_min(eps)
    if clip_update is None or float(clip_update) <= 0.0:
        coef = torch.ones((), device=delta_vec.device, dtype=delta_vec.dtype)
        return delta_vec, norm, coef

    clip_t = torch.tensor(float(clip_update), device=delta_vec.device, dtype=delta_vec.dtype)
    coef = torch.clamp(clip_t / norm, max=1.0)
    return delta_vec * coef, norm, coef


@torch.no_grad()
def eval_with_params(
    net: nn.Module,
    params: Dict[str, torch.Tensor],
    data_loader: DataLoader,
    device: torch.device,
    ce: nn.Module,
):
    net.eval()
    losses = []
    correct = 0
    total = 0
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = functional_call(net, params, (xb,))
        loss = ce(logits, yb)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return (float(np.mean(losses)) if losses else float("nan"), correct / max(total, 1))


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for b in loader:
            yield b


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()

    # data / backbone
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--preprocess_dir", type=str, default="artifacts/preprocess")

    ap.add_argument("--mlp_width", type=int, default=128)
    ap.add_argument("--mlp_layers", type=int, default=5)
    ap.add_argument("--activation", type=str, default="relu")

    # learned optimizer
    ap.add_argument("--opt_hidden_sz", type=int, default=20)
    ap.add_argument("--opt_lr", type=float, default=1e-3)
    ap.add_argument("--out_mul", type=float, default=1e-3)
    ap.add_argument("--no_preproc", action="store_true")
    ap.add_argument("--preproc_factor", type=float, default=10.0)
    ap.add_argument("--detach_grad_input", action="store_true",
                    help="Drop dashed edges (first-order): detach gradient input to optimizer (recommended to match paper).")

    # route B: truncated BPTT
    ap.add_argument("--unroll_steps", type=int, default=20,
                    help="K: truncated BPTT unroll length (paper uses truncated BPTT; e.g., 20).")
    ap.add_argument("--wt", type=float, default=1.0,
                    help="Trajectory weight w_t (default 1.0, matches paper's experiments).")
    ap.add_argument("--meta_val_coef", type=float, default=0.0,
                    help="Optional B2-style: meta_loss += meta_val_coef * val_loss(theta_K). Default 0 (paper-like).")

    # stability knobs
    ap.add_argument("--clip_update", type=float, default=0.0,
                    help="Global-norm clip for Δθ. Default 0 (disabled; not required by paper).")
    ap.add_argument("--reg_update", type=float, default=0.0,
                    help="L2 regularization on raw update output (engineering; default 0).")
    ap.add_argument("--clip_phi_grad", type=float, default=1.0,
                    help="Gradient clipping for optimizer parameters (outer loop).")
    ap.add_argument("--reset_state_each_epoch", action="store_true",
                    help="Reset LSTM hidden/cell to zeros at each epoch start (often helps single-task stability).")
    ap.add_argument("--state_reset_interval", type=int, default=0,
                    help="If >0, reset hidden/cell every N inner steps.")

    # init options (paper mentions IID Gaussian init for optimizee)
    ap.add_argument("--init_normal_std", type=float, default=0.0,
                    help="If >0, re-init optimizee parameters ~ N(0, std). Default 0 keeps module default init.")

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="l2o-mnist1d")
    ap.add_argument("--wandb_group", type=str, default="mnist1d_l2o_trunc_bptt")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    set_seed(args.seed)

    run_name = (
        f"mnist1d_mlp_l2o_truncbptt_data{args.data_seed}_seed{args.seed}_"
        f"L{args.mlp_layers}_W{args.mlp_width}_"
        f"K{args.unroll_steps}_lr{args.opt_lr}_out{args.out_mul}_"
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

    # ---------------- data ----------------
    (xtr, ytr), (xte, yte) = load_mnist1d(length=args.length, seed=args.data_seed)

    xtr = to_NL(xtr, length=args.length).astype(np.float32, copy=False)
    xte = to_NL(xte, length=args.length).astype(np.float32, copy=False)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    split, norm = load_artifacts(Path(args.preprocess_dir), seed=args.data_seed)
    mean = np.array(norm["mean"], np.float32)
    std = np.array(norm["std"], np.float32)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)

    x_train = apply_norm_np(xtr[train_idx], mean, std)
    y_train = ytr[train_idx]
    x_val = apply_norm_np(xtr[val_idx], mean, std)
    y_val = ytr[val_idx]
    x_test = apply_norm_np(xte, mean, std)
    y_test = yte

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )

    val_iter = infinite_loader(val_loader)
    val_eval_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=512, shuffle=False)
    test_eval_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=512, shuffle=False)

    # ---------------- models ----------------
    net = build_mlp(
        input_len=args.length,
        num_layers=args.mlp_layers,
        width=args.mlp_width,
        activation=args.activation,
    ).to(device)

    if args.init_normal_std and args.init_normal_std > 0:
        with torch.no_grad():
            for p in net.parameters():
                p.normal_(mean=0.0, std=float(args.init_normal_std))

    opt_net = LearnedOptimizer(
        hidden_sz=args.opt_hidden_sz,
        preproc=(not args.no_preproc),
        preproc_factor=args.preproc_factor,
        detach_grad_input=bool(args.detach_grad_input),
    ).to(device)

    meta_opt = torch.optim.Adam(opt_net.parameters(), lr=args.opt_lr)
    ce = nn.CrossEntropyLoss()

    # optimizee params (functional) and optimizer state
    params_dict = named_params_dict(net)
    # Ensure params are on device and require grad (functional_call needs tensors with grad)
    params_dict = {k: v.detach().to(device).requires_grad_(True) for k, v in params_dict.items()}
    params_list_template = [params_dict[k] for k in params_dict.keys()]
    n_params = int(sum(p.numel() for p in params_list_template))

    hidden = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]
    cell = [torch.zeros(n_params, args.opt_hidden_sz, device=device) for _ in range(2)]

    # ---------------- logs ----------------
    curve_logger = BatchLossLogger(
        run_dir,
        meta={"method": "l2o_lstm_trunc_bptt", "seed": args.seed, "opt": "lstm_opt", "lr": args.opt_lr},
        filename="curve_l2o_truncbptt.csv",
    )
    mech_path = run_dir / "mechanism_l2o_truncbptt.csv"
    train_log_path = run_dir / "train_log_l2o_truncbptt.csv"
    result_path = run_dir / "result_l2o_truncbptt.json"

    with open(mech_path, "w", encoding="utf-8") as f:
        f.write("global_step,epoch,seg_id,inner_steps,meta_loss,train_loss_avg,train_acc_avg,upd_norm,clip_coef\n")
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,elapsed_sec,train_loss,train_acc,val_loss,test_loss,val_acc,test_acc\n")

    global_step = 0
    start_time = time.time()

    # ---------------- training loop (Route B: truncated BPTT) ----------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        curve_logger.on_epoch_begin(epoch)

        net.train()
        opt_net.train()

        if args.reset_state_each_epoch:
            hidden = [torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])]
            cell = [torch.zeros_like(cell[0]), torch.zeros_like(cell[1])]

        # For epoch metrics, we track actual losses/acc over seen batches (detached)
        epoch_train_loss_sum = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_batches = 0

        seg_id = 0
        train_iter = iter(train_loader)

        while True:
            # Collect K batches for this unroll; if not enough batches left, end epoch.
            batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            try:
                for _ in range(int(args.unroll_steps)):
                    xb, yb = next(train_iter)
                    batches.append((xb, yb))
            except StopIteration:
                break

            if args.state_reset_interval and args.state_reset_interval > 0:
                if global_step % int(args.state_reset_interval) == 0 and global_step > 0:
                    hidden = [torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])]
                    cell = [torch.zeros_like(cell[0]), torch.zeros_like(cell[1])]

            # Build unrolled graph
            meta_loss = torch.zeros((), device=device)
            seg_loss_sum_det = 0.0
            seg_correct = 0
            seg_total = 0

            # We'll update params_dict functionally each inner step
            for inner_t, (xb, yb) in enumerate(batches, start=1):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = functional_call(net, params_dict, (xb,))
                loss = ce(logits, yb)

                # Eq.(3): add w_t * f(theta_t); default w_t=1
                meta_loss = meta_loss + float(args.wt) * loss

                # detached metrics
                with torch.no_grad():
                    seg_loss_sum_det += float(loss.item())
                    pred = logits.argmax(dim=1)
                    seg_correct += int((pred == yb).sum().item())
                    seg_total += int(yb.size(0))

                # Compute optimizee gradient wrt current theta_t
                # (If detach_grad_input=True, optimizer input won't backprop into this grad -> no 2nd derivatives)
                params_list = [params_dict[k] for k in params_dict.keys()]
                grads = torch.autograd.grad(loss, params_list, create_graph=False, retain_graph=False, allow_unused=False)
                g_vec = grads_to_vector(list(grads), params_list)  # [N,1]

                # Learned optimizer proposes update
                upd_vec, h_new, c_new = opt_net(g_vec, hidden, cell)
                delta_vec = upd_vec * float(args.out_mul)  # Δθ proposal (paper: θ_{t+1} = θ_t + g_t)

                # Optional global-norm clip for stability (not required by paper)
                delta_vec_clipped, _, coef = clip_delta_vec(delta_vec, float(args.clip_update) if args.clip_update else 0.0)
                delta_list = vector_to_params(delta_vec_clipped, params_list)

                # Functional parameter update: theta_{t+1} = theta_t + delta
                new_params_dict: Dict[str, torch.Tensor] = {}
                for (name, p), dp in zip(params_dict.items(), delta_list):
                    new_params_dict[name] = p + dp
                params_dict = new_params_dict

                hidden = list(h_new)
                cell = list(c_new)
                global_step += 1

            # Optional B2-style val loss at theta_K (default OFF)
            if args.meta_val_coef and float(args.meta_val_coef) > 0.0:
                xv, yv = next(val_iter)
                xv = xv.to(device)
                yv = yv.to(device)
                val_logits = functional_call(net, params_dict, (xv,))
                val_loss_last = ce(val_logits, yv)
                meta_loss = meta_loss + float(args.meta_val_coef) * val_loss_last

            # Optional regularization on raw update output magnitude (engineering)
            if args.reg_update and float(args.reg_update) > 0.0:
                meta_loss = meta_loss + float(args.reg_update) * torch.mean(upd_vec.pow(2))

            # Outer update (optimize phi)
            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()

            if args.clip_phi_grad and float(args.clip_phi_grad) > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_net.parameters(), max_norm=float(args.clip_phi_grad))
            meta_opt.step()

            # Truncation: detach optimizee params + optimizer states to cut graph
            params_dict = {k: v.detach().requires_grad_(True) for k, v in params_dict.items()}
            hidden = [h.detach() for h in hidden]
            cell = [c.detach() for c in cell]

            # Log segment metrics
            seg_loss_avg = seg_loss_sum_det / max(len(batches), 1)
            seg_acc_avg = seg_correct / max(seg_total, 1)

            epoch_train_loss_sum += seg_loss_sum_det
            epoch_train_correct += seg_correct
            epoch_train_total += seg_total
            epoch_batches += len(batches)

            curve_logger.on_train_step_end(seg_loss_avg, seg_acc_avg)

            with open(mech_path, "a", encoding="utf-8") as f:
                # We can approximate update norm from last step's delta_vec_clipped
                with torch.no_grad():
                    upd_norm = torch.linalg.vector_norm(delta_vec_clipped.view(-1)).clamp_min(1e-12).item()
                    clip_coef = float(coef.detach().item())
                f.write(
                    f"{global_step},{epoch},{seg_id},{len(batches)},"
                    f"{float(meta_loss.item()):.8f},{seg_loss_avg:.8f},{seg_acc_avg:.6f},"
                    f"{upd_norm:.6g},{clip_coef:.6g}\n"
                )

            seg_id += 1

        # ---- epoch eval (using current params_dict) ----
        train_loss_epoch = epoch_train_loss_sum / max(epoch_batches, 1)
        train_acc_epoch = epoch_train_correct / max(epoch_train_total, 1)

        val_loss_epoch, val_acc = eval_with_params(net, params_dict, val_eval_loader, device, ce)
        test_loss_epoch, test_acc = eval_with_params(net, params_dict, test_eval_loader, device, ce)

        total_elapsed = time.time() - start_time
        epoch_elapsed = time.time() - epoch_start

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{total_elapsed:.3f},"
                f"{train_loss_epoch:.8f},{train_acc_epoch:.6f},"
                f"{val_loss_epoch:.8f},{test_loss_epoch:.8f},"
                f"{val_acc:.6f},{test_acc:.6f}\n"
            )

        print(
            f"[MNIST1D-MLP-L2O-TRUNC-BPTT EPOCH {epoch}] "
            f"time={epoch_elapsed:.2f}s total={total_elapsed/60:.2f}min "
            f"train={train_loss_epoch:.4f}/{train_acc_epoch:.4f} "
            f"val={val_loss_epoch:.4f}/{val_acc:.4f} "
            f"test={test_loss_epoch:.4f}/{test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "val_loss": val_loss_epoch,
                    "val_acc": val_acc,
                    "test_loss": test_loss_epoch,
                    "test_acc": test_acc,
                    "time/epoch_sec": epoch_elapsed,
                    "time/total_sec": total_elapsed,
                    "meta/unroll_steps": int(args.unroll_steps),
                }
            )

    curve_logger.on_train_end()
    total_time = time.time() - start_time

    # final test
    final_test_loss, final_test_acc = eval_with_params(net, params_dict, test_eval_loader, device, ce)

    print(
        f"[RESULT-MNIST1D-MLP-L2O-TRUNC-BPTT] TestAcc={final_test_acc:.4f} "
        f"TestLoss={final_test_loss:.4f} (Total time={total_time/60:.2f} min)"
    )

    result = {
        "dataset": f"MNIST-1D(len={args.length})",
        "backbone": "MLP",
        "method": "l2o_coordinatewise_lstm_truncated_bptt",
        "epochs": int(args.epochs),
        "bs": int(args.bs),
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "unroll_steps": int(args.unroll_steps),
        "wt": float(args.wt),
        "meta_val_coef": float(args.meta_val_coef),
        "detach_grad_input": bool(args.detach_grad_input),
        "test_acc": float(final_test_acc),
        "test_loss": float(final_test_loss),
        "elapsed_sec": float(total_time),
        "run_dir": str(run_dir),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["final_test_acc"] = float(final_test_acc)
        wandb_run.summary["final_test_loss"] = float(final_test_loss)
        wandb_run.summary["total_time_sec"] = float(total_time)
        for p in [curve_logger.curve_path, mech_path, train_log_path, result_path]:
            if Path(p).exists():
                wandb.save(str(p))
        wandb_run.finish()


if __name__ == "__main__":
    main()

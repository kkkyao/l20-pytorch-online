# mnist1d_core.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Iterable, Any, Optional
from pathlib import Path
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int, data_seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# MNIST1D preprocessing-compatible helpers
# -------------------------
def to_N40(x: np.ndarray) -> np.ndarray:
    """
    Normalize shape to [N, 40].
    Supports [N,40], [N,40,1], [N,1,40].
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 40:
        return x
    if x.ndim == 3:
        if x.shape[1] == 40 and x.shape[2] == 1:   # [N,40,1] -> [N,40]
            return x[:, :, 0]
        if x.shape[1] == 1 and x.shape[2] == 40:   # [N,1,40] -> [N,40]
            return x[:, 0, :]
    raise AssertionError(f"Unexpected x shape: {x.shape}, expected [N,40] or [N,40,1]/[N,1,40]")


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_npz(preprocess_seed_dir: Path) -> Optional[Path]:
    """
    Prefer *mlp_normed*.npz if present, else any .npz under the directory.
    """
    if not preprocess_seed_dir.exists():
        return None
    cand = sorted(preprocess_seed_dir.glob("*mlp_normed*.npz"))
    if len(cand) > 0:
        return cand[0]
    cand = sorted(preprocess_seed_dir.glob("*.npz"))
    return cand[0] if len(cand) > 0 else None


def load_mnist1d_from_artifacts(
    preprocess_dir: str,
    seed: int,
    val_frac_fallback: float = 0.1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a dict:
      {
        "train": (x_train_norm, y_train),
        "val":   (x_val_norm,   y_val),
        "test":  (x_test_norm,  y_test),
      }
    Priority:
      1) Load exported NPZ under {preprocess_dir}/seed_{seed}/*.npz
      2) Else: load split.json + norm.json and re-generate raw MNIST1D via data_mnist1d.load_mnist1d
    """
    seed_dir = Path(preprocess_dir) / f"seed_{seed}"
    npz_path = _find_npz(seed_dir)

    if npz_path is not None:
        z = np.load(npz_path, allow_pickle=False)
        required = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"NPZ {npz_path} missing keys: {sorted(missing)}")

        x_train = np.asarray(z["x_train"], dtype=np.float32)
        y_train = np.asarray(z["y_train"], dtype=np.int64)
        x_val = np.asarray(z["x_val"], dtype=np.float32)
        y_val = np.asarray(z["y_val"], dtype=np.int64)
        x_test = np.asarray(z["x_test"], dtype=np.float32)
        y_test = np.asarray(z["y_test"], dtype=np.int64)

        # Ensure shapes [N,40]
        x_train = to_N40(x_train)
        x_val = to_N40(x_val)
        x_test = to_N40(x_test)

        return {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (x_test, y_test),
        }

    # Fallback path: use split.json + norm.json + re-load raw MNIST1D
    split_path = seed_dir / "split.json"
    norm_path = seed_dir / "norm.json"
    if not (split_path.exists() and norm_path.exists()):
        raise FileNotFoundError(
            f"Could not find NPZ under {seed_dir} and missing split/norm jsons. "
            f"Expected either *.npz or split.json+norm.json in {seed_dir}."
        )

    split = _load_json(split_path)
    norm = _load_json(norm_path)

    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)

    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    # Load raw MNIST1D (project-specific)
    try:
        from data_mnist1d import load_mnist1d  # type: ignore
    except Exception as e:
        raise ImportError(
            "Fallback loading requires `data_mnist1d.load_mnist1d`. "
            "Could not import it. Either export NPZ via your preprocess script or ensure data_mnist1d.py is importable."
        ) from e

    (x_tr, y_tr), (x_te, y_te) = load_mnist1d(length=40, seed=seed)
    x_tr = to_N40(np.asarray(x_tr)).astype(np.float32, copy=False)
    x_te = to_N40(np.asarray(x_te)).astype(np.float32, copy=False)
    y_tr = np.asarray(y_tr).astype(np.int64, copy=False)
    y_te = np.asarray(y_te).astype(np.int64, copy=False)

    x_tr_split = x_tr[train_idx]
    y_tr_split = y_tr[train_idx]
    x_val_split = x_tr[val_idx]
    y_val_split = y_tr[val_idx]

    x_train_norm = apply_normalization(x_tr_split, mean, std).astype(np.float32, copy=False)
    x_val_norm = apply_normalization(x_val_split, mean, std).astype(np.float32, copy=False)
    x_test_norm = apply_normalization(x_te, mean, std).astype(np.float32, copy=False)

    return {
        "train": (x_train_norm, y_tr_split),
        "val": (x_val_norm, y_val_split),
        "test": (x_test_norm, y_te),
    }


# -------------------------
# Task interface
# -------------------------
@dataclass
class Task:
    """A single (fixed) task = a split with an infinite minibatch iterator."""
    batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]]  # yields (x, y)


class TaskSampler:
    """
    MNIST1D in your pipeline is a *fixed dataset* with stratified train/val split.
    For L2O meta-training we treat each split stream as a 'task' that yields minibatches indefinitely.

    split mapping:
      meta_train -> train
      meta_val   -> val
      meta_test  -> test
    """

    def __init__(
        self,
        data_seed: int,
        preprocess_dir: str = "artifacts/preprocess",
        bs: int = 128,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.data_seed = int(data_seed)
        self.preprocess_dir = str(preprocess_dir)
        self.bs = int(bs)
        self.shuffle = bool(shuffle)
        self.device = device  # batches are moved in runner; keep None here by default

        arrays = load_mnist1d_from_artifacts(preprocess_dir=self.preprocess_dir, seed=self.data_seed)

        self._splits_np: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "train": arrays["train"],
            "val": arrays["val"],
            "test": arrays["test"],
        }

        # Convert once to torch CPU tensors (runner will .to(device))
        self._splits_torch: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k, (x, y) in self._splits_np.items():
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
            if x_t.ndim != 2 or x_t.shape[1] != 40:
                raise AssertionError(f"Split {k} expected x shape [N,40], got {tuple(x_t.shape)}")
            self._splits_torch[k] = (x_t, y_t)

        self.in_dim = 40
        self.num_classes = int(torch.max(self._splits_torch["train"][1]).item() + 1)

        # RNGs for deterministic-but-task-specific shuffles
        self._np_rng = np.random.RandomState(self.data_seed)
        self._torch_gen = torch.Generator(device="cpu")
        self._torch_gen.manual_seed(self.data_seed)

    def _infinite_minibatches(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bs: int,
        shuffle: bool,
        torch_gen: torch.Generator,
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        n = x.shape[0]
        if n == 0:
            raise ValueError("Empty split encountered; cannot create minibatches.")

        while True:
            if shuffle:
                perm = torch.randperm(n, generator=torch_gen)
            else:
                perm = torch.arange(n)

            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                yield x[idx], y[idx]

    def sample_task(self, split: str) -> Task:
        """
        split: 'meta_train' | 'meta_val' | 'meta_test'
        """
        if split == "meta_train":
            key = "train"
        elif split == "meta_val":
            key = "val"
        elif split == "meta_test":
            key = "test"
        else:
            raise ValueError(f"Unknown split={split}. Expected meta_train/meta_val/meta_test.")

        x, y = self._splits_torch[key]

        # Make each Task have its own shuffle stream (deterministic via np_rng)
        task_seed = int(self._np_rng.randint(0, 2**31 - 1))
        tg = torch.Generator(device="cpu")
        tg.manual_seed(task_seed)

        it = self._infinite_minibatches(x, y, bs=self.bs, shuffle=self.shuffle, torch_gen=tg)
        return Task(batch_iter=it)


# -------------------------
# Optimizee (example MLP; adjust if your MNIST1D model differs)
# -------------------------
class OptimizeeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y)


# -------------------------
# Param utilities (named params <-> flat vector)
# -------------------------
@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: torch.Size
    numel: int


def extract_param_specs(model: nn.Module) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    for n, p in model.named_parameters():
        specs.append(ParamSpec(name=n, shape=p.shape, numel=p.numel()))
    return specs


def flatten_params(param_tensors: Dict[str, torch.Tensor], specs: List[ParamSpec]) -> torch.Tensor:
    flats = []
    for s in specs:
        flats.append(param_tensors[s.name].reshape(-1))
    return torch.cat(flats, dim=0)


def unflatten_to_dict(flat: torch.Tensor, specs: List[ParamSpec]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    i = 0
    for s in specs:
        chunk = flat[i:i + s.numel].view(s.shape)
        out[s.name] = chunk
        i += s.numel
    return out


@torch.no_grad()
def init_params_like_model(model: nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone().to(device) for n, p in model.named_parameters()}


# -------------------------
# Gradient preprocessing (paper Appendix A, p=10)
# -------------------------
def preprocess_grad(g: torch.Tensor, p: float = 10.0, eps: float = 1e-8) -> torch.Tensor:
    abs_g = torch.abs(g) + eps
    threshold = math.exp(-p)
    cond = abs_g >= threshold

    feat1 = torch.where(cond, torch.log(abs_g) / p, torch.full_like(g, -1.0))
    feat2 = torch.where(cond, torch.sign(g), math.exp(p) * g)
    return torch.stack([feat1, feat2], dim=-1)


# -------------------------
# Coordinate-wise LSTM learned optimizer
# -------------------------
class CoordWiseLSTM(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 20, out_mul: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_mul = out_mul
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.to_update = nn.Linear(hidden_size, 1)

    def init_state(self, n_coords: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n_coords, self.hidden_size, device=device)
        c = torch.zeros(n_coords, self.hidden_size, device=device)
        return (h, c)

    def step(
        self,
        g_feat: torch.Tensor,  # [N, 2]
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = self.cell(g_feat, state)
        upd = self.to_update(h).squeeze(-1)      # [N]
        upd = torch.tanh(upd) * self.out_mul     # stability
        return upd, (h, c)


# -------------------------
# Inner unroll runner (truncated BPTT)
# -------------------------
@dataclass
class RolloutLog:
    losses: List[float]


def detach_state(state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (state[0].detach(), state[1].detach())


def compute_grads_flat(
    model: nn.Module,
    specs: List[ParamSpec],
    params_dict: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      loss scalar tensor
      grads_flat [N]
    """
    try:
        from torch.nn.utils.stateless import functional_call
    except Exception as e:
        raise RuntimeError(
            "This implementation requires torch.nn.utils.stateless.functional_call. "
            "Please upgrade PyTorch or implement an alternative functional forward."
        ) from e

    logits = functional_call(model, params_dict, (x,))
    loss = loss_fn(logits, y)

    grads = torch.autograd.grad(
        loss,
        list(params_dict.values()),
        create_graph=False,
        retain_graph=False,
    )
    grads_flat = torch.cat([g.reshape(-1) for g in grads], dim=0)
    return loss, grads_flat


def run_unroll(
    task: Task,
    model: nn.Module,
    specs: List[ParamSpec],
    params_flat: torch.Tensor,  # [N]
    lopt: CoordWiseLSTM,
    state: Tuple[torch.Tensor, torch.Tensor],
    steps: int,
    bs: int,  # kept for API compatibility; actual bs is controlled by TaskSampler.bs
    preprocess_p: float,
    device: torch.device,
    ignore_second_order: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], RolloutLog]:
    meta_loss = torch.zeros((), device=device)
    losses_log: List[float] = []

    batch_it = iter(task.batch_iter)
    for _ in range(steps):
        x, y = next(batch_it)
        x = x.to(device)
        y = y.to(device)

        params_dict = unflatten_to_dict(params_flat, specs)
        loss, grads_flat = compute_grads_flat(model, specs, params_dict, x, y)

        losses_log.append(float(loss.detach().cpu()))
        meta_loss = meta_loss + loss

        if ignore_second_order:
            grads_flat = grads_flat.detach()

        g_feat = preprocess_grad(grads_flat, p=preprocess_p)
        update_flat, state = lopt.step(g_feat, state)

        # params_flat keeps graph wrt phi via update_flat
        params_flat = params_flat + update_flat

    return meta_loss, params_flat, state, RolloutLog(losses=losses_log)


# -------------------------
# Generic evaluation runner (pluggable update rule)
# -------------------------
UpdateFn = Callable[[torch.Tensor, torch.Tensor, Any], Tuple[torch.Tensor, Any]]
# signature: (params_flat, grads_flat, state) -> (new_params_flat, new_state)


def rollout_with_update_fn(
    task: Task,
    model: nn.Module,
    specs: List[ParamSpec],
    params_flat0: torch.Tensor,
    steps: int,
    preprocess_p: float,  # kept for convenience; may be used inside update_fn
    device: torch.device,
    update_fn: UpdateFn,
    state0: Any,
    ignore_second_order: bool = True,
) -> RolloutLog:
    batch_it = iter(task.batch_iter)
    params_flat = params_flat0
    state = state0
    losses_log: List[float] = []

    for _ in range(steps):
        x, y = next(batch_it)
        x = x.to(device)
        y = y.to(device)

        params_dict = unflatten_to_dict(params_flat, specs)
        loss, grads_flat = compute_grads_flat(model, specs, params_dict, x, y)
        losses_log.append(float(loss.detach().cpu()))

        if ignore_second_order:
            grads_flat = grads_flat.detach()

        params_flat, state = update_fn(params_flat, grads_flat, state)

    return RolloutLog(losses=losses_log)

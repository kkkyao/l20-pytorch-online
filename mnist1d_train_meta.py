#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-train MNIST1D learned optimizer (coordinate-wise LSTM) with truncated BPTT.

- Loads MNIST1D normalized splits from artifacts (preferred: exported NPZ).
- Trains CoordWiseLSTM parameters (phi) using outer Adam.
- Inner optimization uses functional_call + autograd.grad (no in-place .backward on optimizee params).
- Uses truncated BPTT: unroll K, then detach optimizee params and LSTM state.
- Ignores second-order terms by detaching grads before feeding into LSTM (paper-style).

Artifacts:
  - checkpoints/mnist1d/best.pt
  - checkpoints/mnist1d/epoch_{i}.pt
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Optional

import torch

from mnist1d_core import (
    set_seed,
    TaskSampler,
    OptimizeeMLP,
    extract_param_specs,
    init_params_like_model,
    flatten_params,
    CoordWiseLSTM,
    run_unroll,
    detach_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Reproducibility / data
    p.add_argument("--seed", type=int, default=0, help="torch/random seed")
    p.add_argument("--data_seed", type=int, default=42, help="seed used for MNIST1D loading/split artifacts")
    p.add_argument(
        "--preprocess_dir",
        type=str,
        default="artifacts/preprocess",
        help="directory containing seed_{data_seed}/ with NPZ or split/norm jsons",
    )

    # Meta-training schedule
    p.add_argument("--meta_epochs", type=int, default=10)
    p.add_argument("--tasks_per_epoch", type=int, default=5, help="number of tasks sampled per epoch")
    p.add_argument("--val_tasks", type=int, default=3, help="number of meta-val tasks per epoch for early stopping")

    # Inner loop (optimizee training)
    p.add_argument("--optim_steps", type=int, default=100, help="T: total inner optimization steps per task")
    p.add_argument("--unroll", type=int, default=20, help="K: truncated BPTT unroll length")
    p.add_argument("--bs", type=int, default=128, help="minibatch size for MNIST1D stream")

    # Optimizee model
    p.add_argument("--hidden", type=int, default=20, help="optimizee MLP hidden units (paper MNIST uses 20)")

    # Learned optimizer
    p.add_argument("--lstm_hidden", type=int, default=20, help="LSTM hidden size (paper uses 20)")
    p.add_argument("--out_mul", type=float, default=0.1, help="output multiplier (paper MNIST uses 0.1)")
    p.add_argument("--preprocess_p", type=float, default=10.0, help="gradient preprocessing parameter p (paper uses 10)")
    p.add_argument("--phi_lr", type=float, default=3e-3, help="outer Adam lr for phi")

    # Runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/mnist1d")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--ignore_second_order", action="store_true", default=True)

    # Debug / quick sanity
    p.add_argument("--max_unroll_updates_per_task", type=int, default=-1,
                   help="if >0, limit number of unroll updates per task (for smoke tests)")

    return p.parse_args()


def _evaluate_meta_val(
    sampler: TaskSampler,
    device: torch.device,
    lopt: CoordWiseLSTM,
    hidden: int,
    optim_steps: int,
    unroll: int,
    val_tasks: int,
    preprocess_p: float,
) -> float:
    """
    Evaluate by running a full inner rollout on meta_val tasks (no outer grads).
    Returns mean total loss (sum over steps).
    """
    lopt.eval()
    totals = []

    # Need n_coords: depends on optimizee architecture
    # Use one template optimizee to compute specs.
    in_dim = sampler.in_dim
    out_dim = sampler.num_classes
    model_t = OptimizeeMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    specs_t = extract_param_specs(model_t)
    n_coords = sum(s.numel for s in specs_t)

    with torch.no_grad():
        for _ in range(val_tasks):
            task = sampler.sample_task("meta_val")

            model = OptimizeeMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
            specs = extract_param_specs(model)
            params0 = init_params_like_model(model, device)
            params_flat = flatten_params(params0, specs)
            state = lopt.init_state(n_coords=n_coords, device=device)

            steps_left = optim_steps
            total = 0.0
            while steps_left > 0:
                k = min(unroll, steps_left)
                meta_loss, params_flat, state, _ = run_unroll(
                    task=task,
                    model=model,
                    specs=specs,
                    params_flat=params_flat,
                    lopt=lopt,
                    state=state,
                    steps=k,
                    bs=sampler.bs,  # informational; sampler controls actual batch size
                    preprocess_p=preprocess_p,
                    device=device,
                    ignore_second_order=True,
                )
                total += float(meta_loss.detach().cpu())
                params_flat = params_flat.detach()
                state = detach_state(state)
                steps_left -= k

            totals.append(total)

    return sum(totals) / max(1, len(totals))


def main() -> None:
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    set_seed(args.seed, args.data_seed)
    device = torch.device(args.device)

    # Data/task stream
    sampler = TaskSampler(
        data_seed=args.data_seed,
        preprocess_dir=args.preprocess_dir,
        bs=args.bs,
        shuffle=True,
    )

    in_dim = sampler.in_dim            # 40
    out_dim = sampler.num_classes      # inferred from labels

    # Template optimizee to define parameter specs / coordinate count
    model_template = OptimizeeMLP(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim).to(device)
    specs_template = extract_param_specs(model_template)
    n_coords = sum(s.numel for s in specs_template)

    # Learned optimizer (phi)
    lopt = CoordWiseLSTM(
        input_size=2,
        hidden_size=args.lstm_hidden,
        out_mul=args.out_mul,
    ).to(device)

    outer_optim = torch.optim.Adam(lopt.parameters(), lr=args.phi_lr)

    best_val = float("inf")
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    print("==== MNIST1D Meta-Training ====")
    print(f"device={device}")
    print(f"data_seed={args.data_seed} preprocess_dir={args.preprocess_dir}")
    print(f"in_dim={in_dim} out_dim={out_dim} optimizee_hidden={args.hidden}")
    print(f"T={args.optim_steps} unroll={args.unroll} bs={args.bs}")
    print(f"lstm_hidden={args.lstm_hidden} out_mul={args.out_mul} preprocess_p={args.preprocess_p}")
    print(f"phi_lr={args.phi_lr} meta_epochs={args.meta_epochs} tasks_per_epoch={args.tasks_per_epoch}")
    print(f"n_coords={n_coords}")

    for epoch in range(args.meta_epochs):
        lopt.train()
        t0 = time.time()
        train_meta_losses = []

        for task_i in range(args.tasks_per_epoch):
            task = sampler.sample_task("meta_train")

            # Fresh optimizee initialization per task
            model = OptimizeeMLP(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim).to(device)
            specs = extract_param_specs(model)
            params0 = init_params_like_model(model, device)
            params_flat = flatten_params(params0, specs)

            # Fresh LSTM state per task
            state = lopt.init_state(n_coords=n_coords, device=device)

            steps_left = args.optim_steps
            unroll_updates = 0

            while steps_left > 0:
                k = min(args.unroll, steps_left)

                meta_loss, params_flat, state, _ = run_unroll(
                    task=task,
                    model=model,
                    specs=specs,
                    params_flat=params_flat,
                    lopt=lopt,
                    state=state,
                    steps=k,
                    bs=sampler.bs,
                    preprocess_p=args.preprocess_p,
                    device=device,
                    ignore_second_order=args.ignore_second_order,
                )

                outer_optim.zero_grad(set_to_none=True)
                meta_loss.backward()

                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(lopt.parameters(), args.clip_grad)

                outer_optim.step()

                # Truncate BPTT graph
                params_flat = params_flat.detach()
                state = detach_state(state)

                train_meta_losses.append(float(meta_loss.detach().cpu()))

                steps_left -= k
                unroll_updates += 1
                if args.max_unroll_updates_per_task > 0 and unroll_updates >= args.max_unroll_updates_per_task:
                    break

        # Validation (early stopping on mean total val loss)
        val_mean = _evaluate_meta_val(
            sampler=sampler,
            device=device,
            lopt=lopt,
            hidden=args.hidden,
            optim_steps=args.optim_steps,
            unroll=args.unroll,
            val_tasks=args.val_tasks,
            preprocess_p=args.preprocess_p,
        )

        train_mean = sum(train_meta_losses) / max(1, len(train_meta_losses))
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] train_meta_loss={train_mean:.6f}  val_total_loss={val_mean:.6f}  time={dt:.1f}s")

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "lopt": lopt.state_dict(),
            "outer_optim": outer_optim.state_dict(),
            "best_val": best_val,
            "val_mean": val_mean,
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, f"epoch_{epoch}.pt"))

        if val_mean < best_val:
            best_val = val_mean
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"  -> new best saved: {best_path} (best_val={best_val:.6f})")


if __name__ == "__main__":
    main()

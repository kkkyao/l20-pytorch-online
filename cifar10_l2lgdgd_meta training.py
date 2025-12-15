#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-training loop for CIFAR-10 L2L-GDGD (paper-aligned).

Implements:
- Task = one CIFAR-10 training run
- Meta-loss = sum of optimizee losses along trajectory
- Truncated BPTT with unroll
- Dual-LSTM learned optimizer (conv / fc)
- Explicit meta-eval mode
- Weights & Biases logging
"""

import copy
import numpy as np
import torch
import torch.optim as optim

import wandb

from l2l_cifar10_optimizee import CIFAR10Optimizee, CIFAR10Task
from l2l_cifar10_dual_lstm_optimizer import DualLSTMLearnedOptimizer, detach_var


# ---------------------- run one task ----------------------
def run_one_task(
    opt_net,
    meta_opt,
    optim_steps=100,
    unroll=20,
    out_mul=0.1,
    training=True,
    device="cuda",
    log_prefix=None,
    global_step_offset=0,
):
    """
    Run one optimizee trajectory.

    Returns:
        losses_ever: list of optimizee losses (per step)
    """

    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1  # IMPORTANT: no truncation in eval

    task = CIFAR10Task(training=training)
    optimizee = CIFAR10Optimizee().to(device)

    param_info = optimizee.all_named_parameters()
    n_params = sum(int(np.prod(p.size())) for _, _, p in param_info)

    hidden, cell = opt_net.init_state(n_params, device)

    if training and meta_opt is not None:
        meta_opt.zero_grad()

    total_meta_loss = None
    losses_ever = []

    for step in range(1, optim_steps + 1):
        loss = optimizee(task)
        losses_ever.append(loss.item())

        total_meta_loss = loss if total_meta_loss is None else total_meta_loss + loss
        loss.backward(retain_graph=training)

        # -------- log per-step loss --------
        if log_prefix is not None:
            wandb.log(
                {
                    f"{log_prefix}/loss_step": loss.item(),
                    f"{log_prefix}/step": step,
                },
                step=global_step_offset + step,
            )

        offset = 0
        new_params = {}
        new_hidden = [torch.zeros_like(h) for h in hidden]
        new_cell = [torch.zeros_like(c) for c in cell]

        for param_type, name, p in param_info:
            sz = int(np.prod(p.size()))
            grad = detach_var(p.grad.view(sz, 1))

            update, h_new, c_new = opt_net.step(
                param_type=param_type,
                grad=grad,
                hidden=[h[offset:offset + sz] for h in hidden],
                cell=[c[offset:offset + sz] for c in cell],
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
                meta_opt.zero_grad()
                total_meta_loss.backward()
                meta_opt.step()

            optimizee = CIFAR10Optimizee(
                params={k: detach_var(v) for k, v in new_params.items()}
            ).to(device)

            hidden = [detach_var(h) for h in new_hidden]
            cell = [detach_var(c) for c in new_cell]
            total_meta_loss = None
        else:
            optimizee = CIFAR10Optimizee(params=new_params).to(device)
            hidden = new_hidden
            cell = new_cell

    return losses_ever


# ---------------------- meta-training entry ----------------------
def meta_train_cifar10(
    meta_epochs=50,
    optim_steps=100,
    unroll=20,
    lr=1e-3,
    device="cuda",
):
    """
    Full meta-training loop for CIFAR-10 L2L-GDGD.
    """

    wandb.init(
        project="cifar10-l2lgdgd",
        name="dual_lstm_meta_training",
        config={
            "meta_epochs": meta_epochs,
            "optim_steps": optim_steps,
            "unroll": unroll,
            "lr": lr,
            "optimizer": "Dual-LSTM",
        },
    )

    opt_net = DualLSTMLearnedOptimizer().to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    global_step = 0

    for ep in range(meta_epochs):
        # ---------------- meta-train ----------------
        for _ in range(20):
            run_one_task(
                opt_net,
                meta_opt,
                optim_steps=optim_steps,
                unroll=unroll,
                training=True,
                device=device,
                log_prefix="meta/train",
                global_step_offset=global_step,
            )
            global_step += optim_steps

        # ---------------- meta-eval ----------------
        eval_losses = []
        for _ in range(5):
            losses = run_one_task(
                opt_net,
                meta_opt=None,
                optim_steps=optim_steps,
                unroll=unroll,
                training=False,
                device=device,
                log_prefix="meta/eval",
                global_step_offset=global_step,
            )
            eval_losses.append(sum(losses))

        eval_loss = float(np.mean(eval_losses))

        wandb.log(
            {
                "meta/eval_loss_sum": eval_loss,
                "meta/epoch": ep,
            }
        )

        print(f"[MetaEpoch {ep:03d}] eval meta-loss = {eval_loss:.4f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = copy.deepcopy(opt_net.state_dict())
            wandb.run.summary["best_meta_loss"] = best_loss

    wandb.finish()
    return best_loss, best_state


# ---------------------- main ----------------------
if __name__ == "__main__":
    best_loss, best_opt_state = meta_train_cifar10()
    print("Best CIFAR-10 meta-loss:", best_loss)

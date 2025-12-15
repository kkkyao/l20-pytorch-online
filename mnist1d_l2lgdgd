#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST-1D: Learning to Learn by Gradient Descent by Gradient Descent (L2L-GDGD)

- Coordinate-wise learned optimizer (LSTM)
- Optimizee: 1-hidden-layer MLP with sigmoid (paper-aligned)
- Data: MNIST-1D (length = 40)
- Meta-loss: sum of optimizee losses along optimization trajectory
- Truncated BPTT with unroll
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from data_mnist1d import load_mnist1d

USE_CUDA = torch.cuda.is_available()


# ---------------------- utils ----------------------
def w(x):
    return x.cuda() if USE_CUDA else x


def detach_var(v):
    """
    Detach variable from previous graph but keep requires_grad.
    This is CRITICAL for L2L-GDGD.
    """
    out = Variable(v.data, requires_grad=True)
    out.retain_grad()
    return w(out)


# ---------------------- MNIST-1D task ----------------------
class MNIST1DLoss:
    """
    Task = data stream.
    Each task corresponds to one training run
    with its own minibatch sequence.
    """

    def __init__(self, training=True, batch_size=128):
        (xtr, ytr), (xte, yte) = load_mnist1d(length=40, seed=42)

        if training:
            x, y = xtr, ytr
        else:
            x, y = xte, yte

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        self.loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=True,
        )
        self._iter = iter(self.loader)

    def sample(self):
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

    This matches the MNIST experiment in L2L-GDGD,
    except that we use MNIST-1D instead of 28x28 images.
    """

    def __init__(self, hidden_dim=20, params=None):
        super().__init__()
        if params is None:
            self.params = {
                "W1": nn.Parameter(torch.randn(40, hidden_dim) * 0.01),
                "b1": nn.Parameter(torch.zeros(hidden_dim)),
                "W2": nn.Parameter(torch.randn(hidden_dim, 10) * 0.01),
                "b2": nn.Parameter(torch.zeros(10)),
            }
        else:
            self.params = params

    def all_named_parameters(self):
        return list(self.params.items())

    def forward(self, loss_obj: MNIST1DLoss):
        x, y = loss_obj.sample()
        x = w(x)
        y = w(y)

        h = torch.sigmoid(x @ self.params["W1"] + self.params["b1"])
        logits = h @ self.params["W2"] + self.params["b2"]

        return F.cross_entropy(logits, y)


# ---------------------- Learned Optimizer ----------------------
class LearnedOptimizer(nn.Module):
    """
    Coordinate-wise LSTM optimizer (same as paper).
    """

    def __init__(self, hidden_sz=20, preproc=True, preproc_factor=10.0):
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

    def _preprocess(self, g):
        g = g.data
        out = torch.zeros(g.size(0), 2, device=g.device)
        keep = (g.abs() >= self.preproc_threshold).squeeze()

        out[:, 0][keep] = (
            torch.log(g.abs()[keep] + 1e-8) / self.preproc_factor
        ).squeeze()
        out[:, 1][keep] = torch.sign(g[keep]).squeeze()

        out[:, 0][~keep] = -1
        out[:, 1][~keep] = (math.exp(self.preproc_factor) * g[~keep]).squeeze()
        return Variable(out)

    def forward(self, grad, hidden, cell):
        if self.preproc:
            grad = self._preprocess(grad)

        h0, c0 = self.lstm1(grad, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))
        update = self.out(h1)

        return update, (h0, h1), (c0, c1)


# ---------------------- Meta-training ----------------------
def do_fit(
    opt_net,
    meta_opt,
    optim_steps=100,
    unroll=20,
    out_mul=0.1,
    training=True,
):
    if training:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    loss_obj = MNIST1DLoss(training=training)
    optimizee = w(MNIST1DOptimizee())

    # count total parameters
    n_params = sum(
        int(np.prod(p.size())) for _, p in optimizee.all_named_parameters()
    )

    hidden = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)]
    cell = [w(torch.zeros(n_params, opt_net.hidden_sz)) for _ in range(2)]

    if training:
        meta_opt.zero_grad()

    total_loss = None
    losses_ever = []

    for step in range(1, optim_steps + 1):
        loss = optimizee(loss_obj)
        losses_ever.append(loss.item())

        total_loss = loss if total_loss is None else total_loss + loss
        loss.backward(retain_graph=training)

        offset = 0
        new_params = {}
        new_hidden = [w(torch.zeros_like(h)) for h in hidden]
        new_cell = [w(torch.zeros_like(c)) for c in cell]

        for name, p in optimizee.all_named_parameters():
            sz = int(np.prod(p.size()))
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
            if training:
                meta_opt.zero_grad()
                total_loss.backward()
                meta_opt.step()

            optimizee = w(
                MNIST1DOptimizee(
                    params={k: detach_var(v) for k, v in new_params.items()}
                )
            )
            hidden = [detach_var(h) for h in new_hidden]
            cell = [detach_var(c) for c in new_cell]
            total_loss = None
        else:
            optimizee = w(MNIST1DOptimizee(params=new_params))
            hidden = new_hidden
            cell = new_cell

    return losses_ever


# ---------------------- Meta-training entry ----------------------
def train_l2lgdgd(
    meta_epochs=50,
    optim_steps=100,
    unroll=20,
    lr=1e-3,
):
    opt_net = w(LearnedOptimizer())
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None

    for ep in range(meta_epochs):
        for _ in range(20):
            do_fit(
                opt_net,
                meta_opt,
                optim_steps=optim_steps,
                unroll=unroll,
                training=True,
            )

        # evaluation on held-out tasks
        eval_loss = np.mean(
            [
                sum(
                    do_fit(
                        opt_net,
                        meta_opt=None,
                        optim_steps=optim_steps,
                        unroll=unroll,
                        training=False,
                    )
                )
                for _ in range(10)
            ]
        )

        print(f"[MetaEpoch {ep:03d}] eval loss = {eval_loss:.4f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_state


if __name__ == "__main__":
    best_loss, best_opt = train_l2lgdgd()
    print("Best meta-loss:", best_loss)

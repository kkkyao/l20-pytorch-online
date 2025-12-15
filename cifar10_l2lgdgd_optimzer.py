#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-LSTM Learned Optimizer for CIFAR-10 (L2L-GDGD, Paper-aligned)

- Coordinate-wise optimizer
- Separate LSTMs for:
    (1) convolutional parameters
    (2) fully-connected parameters
- Gradient preprocessing (Appendix A of the paper)
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Variable


# ---------------------- utils ----------------------
def detach_var(v):
    """
    Detach variable from previous graph but keep requires_grad.
    CRITICAL for truncated BPTT in L2L-GDGD.
    """
    out = Variable(v.data, requires_grad=True)
    out.retain_grad()
    return out


# ---------------------- Single LSTM optimizer core ----------------------
class CoordinateWiseLSTM(nn.Module):
    """
    One coordinate-wise LSTM optimizer.
    This module is shared across parameters of the same type.
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

    # --------------------------------------------------
    def _preprocess(self, g):
        """
        Gradient preprocessing (Appendix A, L2L-GDGD paper)
        """
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

    # --------------------------------------------------
    def forward(self, grad, hidden, cell):
        """
        grad:  [N, 1]   (one gradient per coordinate)
        hidden: tuple(h0, h1)
        cell:   tuple(c0, c1)
        """
        if self.preproc:
            grad = self._preprocess(grad)

        h0, c0 = self.lstm1(grad, (hidden[0], cell[0]))
        h1, c1 = self.lstm2(h0, (hidden[1], cell[1]))

        update = self.out(h1)
        return update, (h0, h1), (c0, c1)


# ---------------------- Dual-LSTM wrapper ----------------------
class DualLSTMLearnedOptimizer(nn.Module):
    """
    Dual-LSTM learned optimizer (Section 3.3, L2L-GDGD).

    - One LSTM for convolutional parameters
    - One LSTM for fully-connected parameters
    """

    def __init__(self, hidden_sz=20, preproc=True):
        super().__init__()

        self.conv_lstm = CoordinateWiseLSTM(
            hidden_sz=hidden_sz,
            preproc=preproc,
        )
        self.fc_lstm = CoordinateWiseLSTM(
            hidden_sz=hidden_sz,
            preproc=preproc,
        )

        self.hidden_sz = hidden_sz

    # --------------------------------------------------
    def init_state(self, n_params, device):
        """
        Initialize hidden and cell states for n_params coordinates.
        """
        h = [
            torch.zeros(n_params, self.hidden_sz, device=device),
            torch.zeros(n_params, self.hidden_sz, device=device),
        ]
        c = [
            torch.zeros(n_params, self.hidden_sz, device=device),
            torch.zeros(n_params, self.hidden_sz, device=device),
        ]
        return h, c

    # --------------------------------------------------
    def step(
        self,
        param_type,
        grad,
        hidden,
        cell,
    ):
        """
        One optimizer step for a block of parameters.

        param_type: "conv" or "fc"
        grad:  [N, 1]
        hidden: tuple(h0, h1)
        cell:   tuple(c0, c1)
        """

        if param_type == "conv":
            return self.conv_lstm(grad, hidden, cell)
        elif param_type == "fc":
            return self.fc_lstm(grad, hidden, cell)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

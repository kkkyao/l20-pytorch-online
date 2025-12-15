#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 Optimizee (Paper-aligned) for L2L-GDGD

Architecture (Section 3.3, L2L-GDGD paper):
- 3 convolutional layers with ReLU + BatchNorm
- Max pooling
- Fully-connected layer with 32 hidden units
- Final linear classifier (10 classes)

IMPORTANT:
- This is a *functional* optimizee.
- Parameters are passed explicitly and can be replaced at each meta-step.
- BatchNorm running statistics are buffers (not optimized by LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Functional layers ----------------------
def conv2d_forward(x, weight, bias, stride=1, padding=1):
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)


def linear_forward(x, weight, bias):
    return x @ weight.t() + bias


def batchnorm_forward(x, gamma, beta, running_mean, running_var, training, eps=1e-5, momentum=0.1):
    """
    Functional BatchNorm (channel-wise).
    Running statistics are buffers and NOT optimized by LSTM.
    """
    if training:
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)

        running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * var.detach())
    else:
        mean = running_mean
        var = running_var

    x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps)
    return gamma[None, :, None, None] * x_hat + beta[None, :, None, None]


# ---------------------- CIFAR-10 Optimizee ----------------------
class CIFAR10Optimizee(nn.Module):
    """
    Paper-aligned CIFAR-10 CNN optimizee.

    Parameter types:
    - "conv": convolutional parameters
    - "fc": fully-connected parameters

    BatchNorm running stats are stored as buffers and
    should NOT be optimized by the learned optimizer.
    """

    def __init__(self, params=None, bn_stats=None):
        super().__init__()

        if params is None:
            # ----- Convolutional layers -----
            self.params = {
                # Conv1: 3 -> 32
                "conv1.weight": nn.Parameter(torch.randn(32, 3, 3, 3) * 0.01),
                "conv1.bias": nn.Parameter(torch.zeros(32)),
                "bn1.gamma": nn.Parameter(torch.ones(32)),
                "bn1.beta": nn.Parameter(torch.zeros(32)),

                # Conv2: 32 -> 64
                "conv2.weight": nn.Parameter(torch.randn(64, 32, 3, 3) * 0.01),
                "conv2.bias": nn.Parameter(torch.zeros(64)),
                "bn2.gamma": nn.Parameter(torch.ones(64)),
                "bn2.beta": nn.Parameter(torch.zeros(64)),

                # Conv3: 64 -> 128
                "conv3.weight": nn.Parameter(torch.randn(128, 64, 3, 3) * 0.01),
                "conv3.bias": nn.Parameter(torch.zeros(128)),
                "bn3.gamma": nn.Parameter(torch.ones(128)),
                "bn3.beta": nn.Parameter(torch.zeros(128)),

                # Fully connected layers
                "fc1.weight": nn.Parameter(torch.randn(32, 128 * 4 * 4) * 0.01),
                "fc1.bias": nn.Parameter(torch.zeros(32)),
                "fc2.weight": nn.Parameter(torch.randn(10, 32) * 0.01),
                "fc2.bias": nn.Parameter(torch.zeros(10)),
            }
        else:
            self.params = params

        # ----- BatchNorm running statistics (buffers) -----
        if bn_stats is None:
            self.bn_stats = {
                "bn1.mean": torch.zeros(32),
                "bn1.var": torch.ones(32),
                "bn2.mean": torch.zeros(64),
                "bn2.var": torch.ones(64),
                "bn3.mean": torch.zeros(128),
                "bn3.var": torch.ones(128),
            }
        else:
            self.bn_stats = bn_stats

    # --------------------------------------------------
    # Required by L2L-GDGD
    # --------------------------------------------------
    def all_named_parameters(self):
        """
        Return parameters with explicit type tags.
        These tags are REQUIRED for dual-LSTM routing.
        """
        named = []
        for name, p in self.params.items():
            if name.startswith("conv") or name.startswith("bn"):
                named.append(("conv", name, p))
            else:
                named.append(("fc", name, p))
        return named

    # --------------------------------------------------
    def forward(self, x, y, training=True):
        """
        Forward pass.
        x: [N, 3, 32, 32]
        y: [N]
        """

        # ----- Conv block 1 -----
        x = conv2d_forward(x, self.params["conv1.weight"], self.params["conv1.bias"])
        x = batchnorm_forward(
            x,
            self.params["bn1.gamma"],
            self.params["bn1.beta"],
            self.bn_stats["bn1.mean"],
            self.bn_stats["bn1.var"],
            training,
        )
        x = F.relu(x)

        # ----- Conv block 2 -----
        x = conv2d_forward(x, self.params["conv2.weight"], self.params["conv2.bias"])
        x = batchnorm_forward(
            x,
            self.params["bn2.gamma"],
            self.params["bn2.beta"],
            self.bn_stats["bn2.mean"],
            self.bn_stats["bn2.var"],
            training,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 16x16

        # ----- Conv block 3 -----
        x = conv2d_forward(x, self.params["conv3.weight"], self.params["conv3.bias"])
        x = batchnorm_forward(
            x,
            self.params["bn3.gamma"],
            self.params["bn3.beta"],
            self.bn_stats["bn3.mean"],
            self.bn_stats["bn3.var"],
            training,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 8x8

        # ----- Flatten -----
        x = x.view(x.size(0), -1)  # [N, 128*4*4]

        # ----- Fully connected -----
        x = linear_forward(x, self.params["fc1.weight"], self.params["fc1.bias"])
        x = F.relu(x)
        logits = linear_forward(x, self.params["fc2.weight"], self.params["fc2.bias"])

        loss = F.cross_entropy(logits, y)
        return loss

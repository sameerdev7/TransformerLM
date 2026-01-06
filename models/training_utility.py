"""Softmax function for models."""

import torch


def softmax(x, dim):
    """Numerically stable softmax."""
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x_stable = x - max_vals
    exp_vals = torch.exp(x_stable)
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    return exp_vals / sum_exp

"""Training utilities."""

import math
import torch
import numpy as np


def get_batch(dataset, batch_size, context_length, device):
    """Sample a batch from dataset."""
    max_start = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    
    x = np.stack([dataset[i:i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])
    
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    return x, y


def gradient_clipping(parameters, max_norm):
    """Clip gradients by global norm."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return
    
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        for g in gradients:
            g.mul_(clip_coef)


def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, max_iters):
    """Cosine learning rate schedule with warmup."""
    if it < warmup_iters:
        return (it / warmup_iters) * max_lr
    
    if it <= max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(progress * math.pi)) * (max_lr - min_lr)
    
    return min_lr

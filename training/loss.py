"""Loss functions."""

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss.
    
    Args:
        logits: Shape (..., vocab_size)
        targets: Shape (...)
        
    Returns:
        Scalar loss
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Flatten
    batch_shape = logits.shape[:-1]
    log_probs_flat = log_probs.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    # Get target log probs
    batch_indices = torch.arange(log_probs_flat.size(0), device=logits.device)
    target_log_probs = log_probs_flat[batch_indices, targets_flat]
    
    # Reshape and compute mean
    target_log_probs = target_log_probs.view(batch_shape)
    return -target_log_probs.mean()

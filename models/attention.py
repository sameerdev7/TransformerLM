import torch
import math
from .training_utility import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Scaled dot-product attention implementation.
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: torch.Tensor - Query tensor of shape (batch_size, ..., seq_len_q, d_k)
        K: torch.Tensor - Key tensor of shape (batch_size, ..., seq_len_k, d_k)
        V: torch.Tensor - Value tensor of shape (batch_size, ..., seq_len_v, d_v)
        mask: torch.Tensor = None - Optional boolean mask of shape (seq_len_q, seq_len_k)
                                   True means attend, False means don't attend
    
    Returns:
        torch.Tensor - Output of shape (batch_size, ..., seq_len_q, d_v)
    """
    # Get dimensions
    d_k = Q.shape[-1]
    
    # Compute scaled dot-product: Q @ K^T / sqrt(d_k)
    # Q shape: (..., seq_len_q, d_k)
    # K shape: (..., seq_len_k, d_k)
    # K.transpose(-2, -1) shape: (..., d_k, seq_len_k)
    # scores shape: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set scores to negative infinity
        # This will make softmax output 0 for those positions
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax along the last dimension (over keys)
    attention_weights = softmax(scores, dim=-1)
    
    # Handle the case where entire rows are masked (all -inf)
    # In this case, softmax will produce NaN, so we replace with 0
    if mask is not None:
        # If a row is entirely masked, attention_weights will have NaN
        # Replace NaN with 0
        attention_weights = torch.where(torch.isnan(attention_weights), 
                                      torch.zeros_like(attention_weights), 
                                      attention_weights)
    
    # Apply attention to values
    # attention_weights shape: (..., seq_len_q, seq_len_k)
    # V shape: (..., seq_len_v, d_v) where seq_len_v == seq_len_k
    # output shape: (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output

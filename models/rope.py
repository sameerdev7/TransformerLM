import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    
    Applies rotary positional embeddings to input tensors by rotating pairs of dimensions
    based on their position in the sequence.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: float - Θ value for the RoPE
            d_k: int - dimension of query and key vectors
            max_seq_len: int - Maximum sequence length that will be inputted
            device: torch.device | None = None - Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute the frequency values for each dimension pair
        # For RoPE, we work with pairs of dimensions, so d_k should be even
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        # Create frequency values: theta^(-2i/d_k) for i = 0, 1, ..., d_k/2 - 1
        dim_indices = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        freqs = theta ** (-2.0 * dim_indices / d_k)
        
        # Create position indices for the maximum sequence length
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # Compute the angles: position * frequency for each position and frequency
        # Shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)
        
        # Precompute cos and sin values
        # We need to repeat each value twice to match the pairing structure
        # Shape: (max_seq_len, d_k)
        cos_vals = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        sin_vals = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        
        # Register as buffers so they move with the module
        self.register_buffer('cos_cached', cos_vals)
        self.register_buffer('sin_cached', sin_vals)
    
    def _rotate_half(self, x):
        """
        Rotate the last dimension of x by swapping and negating pairs of elements.
        For RoPE, we rotate pairs of dimensions: (x1, x2) -> (-x2, x1)
        """
        # Split into two halves and swap with negation
        x1 = x[..., ::2]  # Even indices (0, 2, 4, ...)
        x2 = x[..., 1::2]  # Odd indices (1, 3, 5, ...)
        
        # Interleave -x2 and x1
        # Stack along a new dimension and flatten
        rotated = torch.stack((-x2, x1), dim=-1)
        return rotated.flatten(start_dim=-2)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and apply rotary positional embedding.
        
        Args:
            x: torch.Tensor - Input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor - Token positions of shape (..., seq_len)
            
        Returns:
            torch.Tensor - Output tensor of the same shape with RoPE applied
        """        
        # Extract cos and sin values for the given positions
        # token_positions shape: (..., seq_len)
        # We need to index into our cached cos/sin tensors
        cos_vals = self.cos_cached[token_positions]  # (..., seq_len, d_k)
        sin_vals = self.sin_cached[token_positions]  # (..., seq_len, d_k)
        
        # Apply RoPE: x * cos + rotate_half(x) * sin
        rotated_x = self._rotate_half(x)
        return x * cos_vals + rotated_x * sin_vals

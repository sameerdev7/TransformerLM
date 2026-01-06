import torch
import torch.nn as nn
from .normalization import RMSNorm
from .multihead_attention import MultiHeadSelfAttention
from .swiglu import SwiGLU
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """
    A pre-norm Transformer block.
    
    Architecture (from bottom to top):
    1. Input tensor
    2. Norm -> Causal Multi-Head Self-Attention w/ RoPE -> Add (residual)
    3. Norm -> Position-Wise Feed-Forward -> Add (residual)
    4. Output tensor
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        """
        Construct the Transformer block.
        
        Args:
            d_model: int - Dimensionality of the Transformer block inputs
            num_heads: int - Number of heads to use in multi-head self-attention
            d_ff: int - Dimensionality of the position-wise feed-forward inner layer
            device: torch.device | None = None - Device to store parameters on
            dtype: torch.dtype | None = None - Data type of parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer normalization for the two sublayers (RMSNorm)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)  # Before attention
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)  # Before feed-forward
        
        # Causal Multi-Head Self-Attention w/ RoPE
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        # Position-Wise Feed-Forward (using SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply the Transformer block transformation.
        
        Args:
            x: torch.Tensor - Input tensor of shape (batch_size, seq_len, d_model)
            rope: RotaryPositionalEmbedding = None - Optional RoPE module
            token_positions: torch.Tensor = None - Token positions for RoPE
            
        Returns:
            torch.Tensor - Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First sublayer: Norm -> Causal Multi-Head Self-Attention w/ RoPE -> Add
        norm1_output = self.ln1(x)
        attn_output = self.attn(norm1_output, rope=rope, token_positions=token_positions)
        x = x + attn_output  # Residual connection
        
        # Second sublayer: Norm -> Position-Wise Feed-Forward -> Add  
        norm2_output = self.ln2(x)
        ffn_output = self.ffn(norm2_output)
        x = x + ffn_output  # Residual connection
        
        return x

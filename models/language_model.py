"""
Complete Transformer language model.
"""

import torch
import torch.nn as nn

from .embedding import Embedding
from .transformer import TransformerBlock
from .normalization import RMSNorm
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .attention import softmax


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Architecture:
    1. Token Embedding
    2. N x Transformer Blocks
    3. Final Norm
    4. Output Projection
    5. (Optional) Softmax
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float = 10000.0,
        device=None, 
        dtype=None
    ):
        """
        Initialize Transformer LM.
        
        Args:
            vocab_size: Vocabulary size
            context_length: Maximum context length 
            d_model: Model dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward inner dimension
            rope_theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embeddings
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.ln_f = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
        # RoPE
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta, 
            d_k=d_k, 
            max_seq_len=context_length, 
            device=device
        )
    
    def forward(self, input_ids: torch.Tensor, apply_softmax: bool = True) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            apply_softmax: Whether to apply softmax to get probabilities
            
        Returns:
            Output probabilities (if apply_softmax=True) or logits
            Shape: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate token positions for RoPE
        token_positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Apply Transformer blocks
        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)
        
        # Final norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        # Apply softmax if requested
        if apply_softmax:
            return softmax(logits, dim=-1)
        else:
            return logits

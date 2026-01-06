import torch
import torch.nn as nn
from .linear import Linear
from .attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype) 
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x, rope=None, token_positions=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Apply RoPE if provided
        if rope is not None and token_positions is not None:
            Q_rope = Q.contiguous().view(batch_size * self.num_heads, seq_len, self.d_k)
            K_rope = K.contiguous().view(batch_size * self.num_heads, seq_len, self.d_k)
            
            token_positions_expanded = (
                token_positions.unsqueeze(1)
                .expand(batch_size, self.num_heads, seq_len)
                .contiguous()
                .view(batch_size * self.num_heads, seq_len)
            )
            
            Q_rope = rope(Q_rope, token_positions_expanded)
            K_rope = rope(K_rope, token_positions_expanded)
            
            Q = Q_rope.view(batch_size, self.num_heads, seq_len, self.d_k)
            K = K_rope.view(batch_size, self.num_heads, seq_len, self.d_k)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        causal_mask = ~causal_mask
        
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(attn_output)

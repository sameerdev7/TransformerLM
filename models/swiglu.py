import torch
import torch.nn as nn
from .linear import Linear


def silu(x):
    """SiLU activation."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        if d_ff is None:
            d_ff = int(8/3 * d_model)
            d_ff = ((d_ff + 31) // 64) * 64
        
        self.d_ff = d_ff
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)  
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.W2(silu(self.W1(x)) * self.W3(x))

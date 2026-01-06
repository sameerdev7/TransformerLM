import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    A linear transformation module that inherits from torch.nn.Module.
    Performs y = xW^T without bias.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: int - final dimension of the input
            out_features: int - final dimension of the output  
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter W (not W^T) for memory ordering reasons
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize weights using truncated normal distribution.
        Linear weights: N(μ=0, σ²=2/(d_in + d_out)) truncated at [-3σ, 3σ]
        """
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return torch.matmul(x, self.W.T)

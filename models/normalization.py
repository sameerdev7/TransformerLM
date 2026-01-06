import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) module.
    
    Rescales each activation a_i as: RMSNorm(a_i) = a_i/RMS(a) * g_i
    where RMS(a) = sqrt(1/d_model * ∑a^2_i + ε)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model: int - Hidden dimension of the model
            eps: float = 1e-5 - Epsilon value for numerical stability
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameter g_i
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and apply RMSNorm.
        
        Args:
            x: torch.Tensor - Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            torch.Tensor - Normalized tensor of the same shape
        """
        # Store original dtype
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS: sqrt(1/d_model * ∑a^2_i + ε)
        # Mean of squares over the last dimension (d_model)
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        
        # Apply RMSNorm: a_i/RMS(a) * g_i
        result = (x / rms) * self.g
        
        # Return in original dtype
        return result.to(in_dtype)

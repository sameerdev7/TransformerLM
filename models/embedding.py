import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    An embedding module that inherits from torch.nn.Module.
    Performs embedding lookup by indexing into an embedding matrix.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.
        
        Args:
            num_embeddings: int - Size of the vocabulary
            embedding_dim: int - Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None - Device to store the parameters on
            dtype: torch.dtype | None = None - Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding matrix parameter with shape (vocab_size, d_model)
        # Store with d_model as the final dimension
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize embedding weights using truncated normal distribution.
        Embedding: N(μ=0, σ²=1) truncated at [-3, 3]
        """
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: torch.Tensor - Token IDs with shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor - Embedding vectors with shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embeddings[token_ids]

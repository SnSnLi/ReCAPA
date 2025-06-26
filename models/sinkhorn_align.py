import torch
from torch import nn, Tensor

try:
    from geomloss import SamplesLoss
except ImportError:
    # This is a fallback so that the code can be imported even if geomloss is not installed.
    # A runtime error will be raised if the module is actually used.
    SamplesLoss = None

class SinkhornAlign(nn.Module):
    """
    Computes a batched Sinkhorn divergence between two sets of embeddings.
    """
    def __init__(self, epsilon: float, scaling: float = 0.9, normalize_embeds: bool = True, backend: str = 'online'):
        """
        Initializes the SinkhornAlign module.
        Args:
            epsilon (float): The regularization strength for the Sinkhorn algorithm.
            scaling (float): The scaling factor for cost matrix stabilization.
            normalize_embeds (bool): Whether to L2-normalize embeddings before computing distance.
            backend (str): The backend to use for SamplesLoss ('tensorized', 'online', 'multiscale').
        """
        super().__init__()
        self.epsilon = epsilon
        self.scaling = scaling
        self.normalize_embeds = normalize_embeds
        if SamplesLoss is not None:
            self.sinkhorn = SamplesLoss(
                "sinkhorn", 
                p=2, 
                blur=self.epsilon, 
                scaling=self.scaling, 
                backend=backend
            )
        else:
            self.sinkhorn = None

    def forward(self, E_p: Tensor, E_t: Tensor) -> Tensor:
        """
        Calculates the Sinkhorn divergence.
        Args:
            E_p (Tensor): Embeddings for the first distribution, shape (B, N, D) or (N, D).
            E_t (Tensor): Embeddings for the second distribution, shape (B, M, D) or (M, D).
        Returns:
            Tensor: The Sinkhorn divergence, a scalar tensor.
        """
        if self.sinkhorn is None:
            raise ImportError("Geomloss is not installed. Please install it to use SinkhornAlign: `pip install geomloss`")

        if self.normalize_embeds:
            E_p = torch.nn.functional.normalize(E_p, p=2, dim=-1)
            E_t = torch.nn.functional.normalize(E_t, p=2, dim=-1)

        # Handle unbatched inputs by adding a batch dimension
        is_batched = E_p.dim() == 3
        if not is_batched:
            E_p = E_p.unsqueeze(0)
            E_t = E_t.unsqueeze(0)

        if E_p.dim() != 3 or E_t.dim() != 3 or E_t.shape[0] != E_p.shape[0]:
            raise ValueError(
                "Input tensors must be 2D (n, d) or 3D (b, n, d), and batch sizes must match."
            )
            
        loss = self.sinkhorn(E_p, E_t)  # returns (B,)

        return loss.mean() 
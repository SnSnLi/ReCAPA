import torch
import torch.nn as nn
from typing import Optional
from .forward_policy_network import PolicyNetwork

class CGFModule(nn.Module):
    """
    Contrastive Gradient Flow (CGF) Module: Maps loss gradients to policy network parameters.
    This module computes the gradient of a given loss with respect to a set of embeddings,
    and then backpropagates these gradients to the policy network that influenced the
    generation of the embeddings.
    """
    def __init__(
        self,
        hidden_dim: int,
        policy_net: Optional[nn.Module] = None
    ):
        super(CGFModule, self).__init__()
        self.policy_net = policy_net
        # Embedding dimensions per level
        self.embedding_dims = {
            'low': hidden_dim,      # Short-term actions (5-10 steps)
            'mid': hidden_dim * 2,  # Mid-term sub-tasks (50-100 steps)
            'high': hidden_dim * 4  # Long-term coordination (~500 steps)
        }

    def set_policy(self, policy_net: nn.Module):
        """
        Set the policy network for parameter updates.
        :param policy_net: Policy network (PolicyNetwork).
        """
        self.policy_net = policy_net

    def forward(
        self,
        loss: torch.Tensor,
        anchor_emb: torch.Tensor,
        level: str = 'low'
    ):
        """
        Map loss gradients to policy network parameters. The gradients are accumulated
        in the .grad attribute of the policy network's parameters, to be used by an external optimizer.
        
        :param loss: The loss tensor to compute gradients from (e.g., total_loss).
        :param anchor_emb: The anchor trajectory embedding from RceModule.
        :param level: HRN level ('low', 'mid', 'high').
        """
        if level not in self.embedding_dims:
            raise ValueError(f"Invalid level: {level}. Expected 'low', 'mid', or 'high'.")

        if self.policy_net is None:
            raise RuntimeError("Policy network not set. Call set_policy.")

        if not anchor_emb.requires_grad:
            return

        # Compute gradients w.r.t. anchor embedding (∇_{h_τ} L)
        emb_grads = torch.autograd.grad(
            loss,
            anchor_emb,
            retain_graph=True,
            create_graph=False, # Create graph is not needed if we are not doing second order optimization
            allow_unused=True
        )[0]

        if emb_grads is None:
            return

        # Backpropagate gradients through the policy network.
        # This will accumulate gradients in the .grad attributes of the policy network parameters.
        if anchor_emb.grad_fn is not None:
            anchor_emb.backward(emb_grads, retain_graph=True)
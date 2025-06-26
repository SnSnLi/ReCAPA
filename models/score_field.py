import torch
from torch import nn, Tensor
from typing import Optional

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ScoreFieldAlign(nn.Module):
    """
    Implements a score matching loss based on a conditional score network.
    The score network s_theta(x, p) approximates the gradient of the log-likelihood
    of the data distribution, grad_x log p(x|p).
    """
    def __init__(self, input_dim: int, hidden_dim: int, prompt_dim: int, sigmas: list[float] = [0.1, 0.5, 1.0], num_layers: int = 3):
        """
        Initializes the ScoreFieldAlign module.
        Args:
            input_dim: The dimension of the state embeddings (x).
            hidden_dim: The hidden dimension of the score network.
            prompt_dim: The dimension of the prompt embeddings (p).
            sigmas: A list of standard deviations for the Gaussian noise for denoising score matching.
            num_layers: The number of layers in the score network MLP.
        """
        super().__init__()
        self.score_net = MLP(
            input_dim=input_dim + prompt_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers
        )
        self.register_buffer('sigmas', torch.tensor(sigmas))

    def forward(self, states: Tensor, prompt_emb: Tensor) -> Tensor:
        """
        Calculates the score matching loss using the denoising score matching objective.
        
        Args:
            states: The trajectory state embeddings, shape (B, D_state) or (B, T, D_state).
            prompt_emb: The prompt embeddings, shape (B, D_prompt) for global prompt,
                        or (B, N_tokens, D_prompt) for token-level prompts.
                        
        Returns:
            A scalar tensor representing the score matching loss.
        """
        # If prompt_emb are token-level, average them to get a global prompt embedding.
        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb.mean(dim=1)

        # Sample sigmas for the batch
        batch_size = states.shape[0]
        indices = torch.randint(0, len(self.sigmas), (batch_size,), device=states.device)
        sigmas = self.sigmas[indices]
        
        # Reshape sigmas for broadcasting, assuming states are (B, ...)
        sigmas_view = sigmas.view(batch_size, *([1] * (states.dim() - 1)))

        # Add Gaussian noise to states for denoising score matching
        noise = torch.randn_like(states) * sigmas_view
        perturbed_states = states + noise
        
        # The ground truth score for a Gaussian perturbation is -noise / sigma^2
        target_score = -noise / (sigmas_view ** 2)
        
        # Prepare prompt for concatenation. It should be expanded if states have a time dimension.
        if states.dim() == 3 and prompt_emb.dim() == 2:
            # states: (B, T, D_state), prompt_emb: (B, D_prompt)
            prompt_emb_expanded = prompt_emb.unsqueeze(1).expand(-1, states.shape[1], -1)
        else:
            prompt_emb_expanded = prompt_emb

        # Predict the score using the network
        score_net_input = torch.cat([perturbed_states, prompt_emb_expanded], dim=-1)
        predicted_score = self.score_net(score_net_input)
        
        # The loss is the mean squared error between the predicted and true scores.
        # We sum over the feature dimension and then average over the batch (and time).
        loss = (predicted_score - target_score).pow(2).sum(dim=-1).mean()
        
        return loss 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    """
    策略网络：将状态映射为动作分布，支持连续和离散动作空间，多智能体版本。
    与GEMR和反思头模块配合使用。
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], continuous=True, num_agents=1):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.num_agents = num_agents
        # 为每个 agent 构建独立的 MLP head
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU()
            ) for _ in range(num_agents)
        ])
        # mean/log_std 或 logits
        if continuous:
            self.mean_heads = nn.ModuleList([
                nn.Linear(hidden_dims[1], action_dim) for _ in range(num_agents)
            ])
            self.log_std = nn.Parameter(torch.zeros(num_agents, action_dim))
        else:
            self.logits_heads = nn.ModuleList([
                nn.Linear(hidden_dims[1], action_dim) for _ in range(num_agents)
            ])

    def forward(self, state, agent_id: int):
        """
        前向传播，返回动作分布参数。
        state: Tensor([state_dim])
        agent_id: int
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]
        features = self.heads[agent_id](state)  # [B, hidden]
        if self.continuous:
            mu = self.mean_heads[agent_id](features)
            std = self.log_std[agent_id].exp().expand_as(mu)
            return mu, std
        else:
            logits = self.logits_heads[agent_id](features)
            return logits

    def get_action_dist(self, state, agent_id: int):
        """
        获取动作分布对象，用于采样和计算log概率。
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        if self.continuous:
            mu, std = self.forward(state, agent_id)
            return Normal(mu, std)
        else:
            logits = self.forward(state, agent_id)
            return Categorical(logits=logits)

    def sample_action(self, state, agent_id: int):
        """
        从动作分布中采样动作，返回 action 和 log_prob。
        """
        dist = self.get_action_dist(state, agent_id)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, states, actions, agent_id: int):
        """
        评估给定动作的 log_prob 和 entropy，用于训练和反思头计算。  
        states: Tensor([B, state_dim])  
        actions: Tensor([B, action_dim] 或 [B])
        """
        dist = self.get_action_dist(states, agent_id)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        if self.continuous:
            log_probs = log_probs.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        return log_probs, entropy



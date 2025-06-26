import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from .rce import RceModule
# from .llm_trajectory import LLMTrajectoryGenerator # No longer needed here
from .cgf import CGFModule
from .forward_policy_network import PolicyNetwork

MultiTransition = namedtuple('MultiTransition', ['agent_id', 'state', 'action', 'reward'])

class HPCRPredictionHead(nn.Module):
    """
    HPCR预测头：实现层级间的预测式对比学习
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        # 构建预测网络 f_θ
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.prediction_network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z_current: torch.Tensor) -> torch.Tensor:
        return self.prediction_network(z_current)
    
    def compute_predictive_infonce_loss(
        self,
        z_current: torch.Tensor,
        z_next_positive: torch.Tensor,
        z_next_negatives: List[torch.Tensor],
        negative_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算预测式InfoNCE损失"""
        z_pred = self.forward(z_current)
        
        # 确保维度匹配
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)
        if z_next_positive.dim() == 1:
            z_next_positive = z_next_positive.unsqueeze(0)
        
        pos_sim = F.cosine_similarity(z_pred, z_next_positive, dim=-1)
        pos_logits = pos_sim / self.temperature
        
        neg_logits_list = []
        for neg_sample in z_next_negatives:
            if neg_sample.dim() == 1:
                neg_sample = neg_sample.unsqueeze(0)
            neg_sim = F.cosine_similarity(z_pred, neg_sample, dim=-1)
            neg_logits_list.append(neg_sim / self.temperature)
        
        if not neg_logits_list:
            return torch.tensor(0.0, device=z_current.device, requires_grad=True)
        
        neg_logits = torch.stack(neg_logits_list, dim=1)
        
        if negative_weights is not None:
            neg_logits = neg_logits * negative_weights.unsqueeze(0)
        
        numerator = torch.exp(pos_logits)
        denominator = numerator + torch.exp(neg_logits).sum(dim=1)
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        return loss.mean()

class ReflectionHead(nn.Module):
    """
    反射头模块：实现 Reflective Contrastive Policy Learning (RCPL) 及其对齐扩展。
    该模块协调 RCE 模块，计算对比损失，并结合 Sinkhorn 和 Score-Field 对齐损失
    来引导策略学习。
    """
    def __init__(
        self,
        rce_module: RceModule,
        cgf_module: CGFModule,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.gamma = gamma
        self.rce_module = rce_module
        self.cgf_module = cgf_module

    def set_policy(self, policy_net):
        """
        设置策略网络。
        """
        self.cgf_module.set_policy_net(policy_net)

    def forward(
        self,
        anchor_traj: torch.Tensor,
        pos_trajs: List[torch.Tensor],
        neg_trajs: List[torch.Tensor],
        prompt_token_embeddings: torch.Tensor,
        prompt_global_embedding: torch.Tensor,
        sinkhorn_weight: float,
        score_field_weight: float,
        neg_weights: torch.Tensor = None,
        level: str = 'low'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        执行完整的反思与对齐过程，返回总损失和各项损失的 breakdown。

        :param anchor_traj: 锚点轨迹, shape (T, input_dim)
        :param pos_trajs: 正样本轨迹列表, each shape (T, input_dim)
        :param neg_trajs: 负样本轨迹列表, each shape (T, input_dim)
        :param prompt_token_embeddings: Prompt 的 token 级别嵌入, shape (N, D_prompt)
        :param prompt_global_embedding: Prompt 的全局嵌入, shape (D_prompt,)
        :param sinkhorn_weight: a float for sinkhorn loss weight
        :param score_field_weight: a float for score field loss weight
        :param neg_weights: 负样本权重, shape (num_neg,).
        :param level: HRN 层次 ('low', 'mid', 'high')
        :return: A tuple containing:
                 - total_loss: 加权后的总损失
                 - loss_breakdown: 包含各项损失详情的字典
        """
        # 1. 计算 InfoNCE 对比损失
        contrastive_loss = self.rce_module.compute_contrastive_loss(
            anchor_traj=anchor_traj,
            pos_trajs=pos_trajs,
            neg_trajs=neg_trajs,
            neg_weights=neg_weights,
            level=level
        )

        # 2. 计算对齐损失
        # 首先，为锚点轨迹获取逐状态的嵌入
        # (T, D_embedding)
        anchor_state_embeddings = self.rce_module.get_trajectory_state_embeddings(anchor_traj, level=level)

        # 2a. Sinkhorn 对齐损失
        # (N, D_prompt) vs (T, D_embedding)
        sinkhorn_loss = self.rce_module.compute_sinkhorn_loss(
            prompt_token_embeddings=prompt_token_embeddings,
            trajectory_state_embeddings=anchor_state_embeddings
        )

        # 2b. 分数场对齐损失
        # (T, D_embedding) vs (D_prompt,)
        score_loss = self.rce_module.compute_score_loss(
            trajectory_state_embeddings=anchor_state_embeddings,
            prompt_embedding=prompt_global_embedding
        )

        # 3. 加权合并总损失
        total_loss = (
            contrastive_loss +
            sinkhorn_weight * sinkhorn_loss +
            score_field_weight * score_loss
        )
        
        # 4. CGF: Map gradients of total loss to the policy network
        self.cgf_module(
            loss=total_loss,
            anchor_emb=anchor_state_embeddings,
            level=level
        )
        
        loss_breakdown = {
            "contrastive_loss": contrastive_loss.item(),
            "sinkhorn_loss": sinkhorn_loss.item(),
            "score_field_loss": score_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, loss_breakdown

    def compute_multi_agent_loss(
        self,
        current_seg: List[MultiTransition],
        candidate_trajs: List[List[MultiTransition]],
        level: str = 'low'
    ) -> torch.Tensor:
        """
        对所有 agent 在指定层次执行 RCPL 反思，累加损失。
        :param current_seg: 当前轨迹片段
        :param candidate_trajs: 候选轨迹列表
        :param level: 层次（'low', 'mid', 'high'）
        :return: 总 InfoNCE 损失
        """
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        for aid in range(self.num_agents):
            # This function signature needs to be updated or refactored
            # to handle deviation points per agent.
            # For now, this part is considered out of scope of the current change.
            # total_loss = total_loss + self.forward(
            #     current_seg, candidate_trajs, aid, level
            # )
            pass
        return total_loss


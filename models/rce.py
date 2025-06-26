import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .sinkhorn_align import SinkhornAlign
from .score_field import ScoreFieldAlign

class RceModule(nn.Module):
    """
    Reflective Contrastive Embedding (RCE) 模块：编码轨迹并计算 InfoNCE 对比损失。
    使用 Transformer 编码器生成嵌入向量，支持 HRN 的多层次轨迹（'low', 'mid', 'high'）。
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        temperature: float = 0.1,
        # Alignment module parameters
        sinkhorn_epsilon: float = 0.1,
        score_field_hidden_dim: int = 128,
        score_field_num_layers: int = 3,
        prompt_embedding_dim: int = 768, # Example dim, adjust as needed
    ):
        super(RceModule, self).__init__()
        self.temperature = temperature
        self.input_dim = state_dim + action_dim + 1  # state + action + reward

        # 多层次嵌入维度配置
        self.embedding_dims = {
            'low': hidden_dim,      # 短期轨迹
            'mid': hidden_dim * 2,  # 中期子任务
            'high': hidden_dim * 4  # 长期协调
        }

        # --- 新增: 输入线性映射层 ---
        self.input_projection = nn.Linear(self.input_dim, hidden_dim)

        # Transformer 编码器（每层独立配置）
        self.encoders = nn.ModuleDict({
            level: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, # <-- 修改: 使用 hidden_dim
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4, # <-- 惯例: FF维度通常是d_model的4倍
                    dropout=0.1,
                    batch_first=True  # <-- 推荐: 使用 batch_first=True
                ),
                num_layers=num_layers
            )
            for level in ['low', 'mid', 'high']
        })

        # 嵌入头（映射到层次特定维度）
        self.embedding_heads = nn.ModuleDict({
            level: nn.Linear(hidden_dim, self.embedding_dims[level]) # <-- 修改: 输入是 hidden_dim
            for level in ['low', 'mid', 'high']
        })

        # --- Alignment Modules ---
        self.sinkhorn_align = SinkhornAlign(epsilon=sinkhorn_epsilon)

        # The input to the score network is the state concatenated with the prompt embedding.
        # We'll use the 'low' level embedding dim for the state representation in the score field.
        score_input_dim = self.embedding_dims['low']
        self.score_field_align = ScoreFieldAlign(
            input_dim=score_input_dim,
            hidden_dim=score_field_hidden_dim,
            prompt_dim=prompt_embedding_dim,
            num_layers=score_field_num_layers,
        )

    def encode_trajectory(self, traj: torch.Tensor, level: str = 'low') -> torch.Tensor:
        """
        使用 Transformer 编码轨迹为嵌入向量。
        :param traj: Tensor, shape (T, input_dim)，轨迹序列
        :param level: 层次（'low', 'mid', 'high'）
        :return: 嵌入向量, shape (embedding_dim,)
        """
        if level not in self.embedding_dims:
            raise ValueError(f"Invalid level: {level}. Expected 'low', 'mid', or 'high'.")

        # 确保输入维度正确
        if traj.dim() == 2:
            traj = traj.unsqueeze(0)  # (1, T, input_dim)
        elif traj.dim() != 3:
            raise ValueError(f"Expected traj shape (T, input_dim) or (1, T, input_dim), got {traj.shape}")

        # Transformer 编码
        projected_traj = self.input_projection(traj) # <-- 新增: 应用输入映射
        encoded = self.encoders[level](projected_traj)  # (B, T, hidden_dim)

        # 平均池化并映射到嵌入空间
        pooled = encoded.mean(dim=1)  # (B, hidden_dim)
        embedding = self.embedding_heads[level](pooled) # (B, embedding_dim)

        if not is_batched:
            embedding = embedding.squeeze(0)

        return embedding

    def get_trajectory_state_embeddings(self, traj: torch.Tensor, level: str = 'low') -> torch.Tensor:
        """
        Encodes a trajectory and returns the embeddings for each state in the trajectory,
        not just the pooled trajectory embedding.
        :param traj: Tensor, shape (T, input_dim) or (B, T, input_dim)
        :param level: The hierarchy level ('low', 'mid', 'high')
        :return: State embeddings, shape (B, T, embedding_dim)
        """
        if level not in self.embedding_dims:
            raise ValueError(f"Invalid level: {level}. Expected 'low', 'mid', or 'high'.")

        is_batched = traj.dim() == 3
        if not is_batched:
            traj = traj.unsqueeze(0)  # (1, T, input_dim)

        # Transformer encoding
        projected_traj = self.input_projection(traj) # <-- 新增: 应用输入映射
        encoded = self.encoders[level](projected_traj)  # (B, T, hidden_dim)
        
        # Project to embedding space for each state
        # The embedding head expects (N, D), so we reshape.
        batch_size, seq_len, _ = encoded.shape
        state_embeddings = self.embedding_heads[level](encoded.view(batch_size * seq_len, -1))
        state_embeddings = state_embeddings.view(batch_size, seq_len, -1) # (B, T, embedding_dim)

        if not is_batched:
            state_embeddings = state_embeddings.squeeze(0)

        return state_embeddings

    def compute_contrastive_loss(
        self,
        anchor_traj: torch.Tensor,
        pos_trajs: List[torch.Tensor],
        neg_trajs: List[torch.Tensor],
        neg_weights: torch.Tensor = None,
        level: str = 'low'
    ) -> torch.Tensor:
        """
        计算 InfoNCE 对比损失，可对负样本进行加权。
        :param anchor_traj: 锚点轨迹, shape (T, input_dim)
        :param pos_trajs: 正样本轨迹列表, each shape (T, input_dim)
        :param neg_trajs: 负样本轨迹列表, each shape (T, input_dim)
        :param neg_weights: 负样本权重, shape (num_neg,). 如果为 None，则权重全为 1。
        :param level: 层次（'low', 'mid', 'high'）
        :return: InfoNCE 损失
        """
        device = anchor_traj.device

        # 编码锚点轨迹
        anchor_emb = self.encode_trajectory(anchor_traj, level)  # (embedding_dim,)

        # 编码正样本和负样本
        pos_embs = [self.encode_trajectory(t, level) for t in pos_trajs if t.numel() > 0]
        neg_embs = [self.encode_trajectory(t, level) for t in neg_trajs if t.numel() > 0]

        # 检查样本有效性
        if not pos_embs or not neg_embs:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算余弦相似度
        pos_sim = torch.stack([
            F.cosine_similarity(anchor_emb, p, dim=-1) for p in pos_embs
        ])  # (num_pos,)
        neg_sim = torch.stack([
            F.cosine_similarity(anchor_emb, n, dim=-1) for n in neg_embs
        ])  # (num_neg,)

        # 如果未提供权重，则默认为 1
        if neg_weights is None:
            neg_weights = torch.ones_like(neg_sim)

        # InfoNCE 损失
        numerator = torch.exp(pos_sim / self.temperature).sum()
        denominator = numerator + (neg_weights * torch.exp(neg_sim / self.temperature)).sum()
        loss = -torch.log(numerator / (denominator + 1e-8))

        return loss

    def compute_sinkhorn_loss(
        self, 
        prompt_token_embeddings: torch.Tensor, 
        trajectory_state_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Sinkhorn divergence between prompt token embeddings and trajectory state embeddings.
        :param prompt_token_embeddings: Embeddings of prompt tokens, shape (B, N, D) or (N, D)
        :param trajectory_state_embeddings: Embeddings of trajectory states, shape (B, M, D) or (M, D)
        :return: Sinkhorn loss scalar tensor.
        """
        return self.sinkhorn_align(prompt_token_embeddings, trajectory_state_embeddings)

    def compute_score_loss(
        self,
        trajectory_state_embeddings: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss.
        :param trajectory_state_embeddings: Embeddings of trajectory states, shape (B, T, D_state)
        :param prompt_embedding: Global prompt embedding, shape (B, D_prompt)
        :return: Score matching loss scalar tensor.
        """
        return self.score_field_align(trajectory_state_embeddings, prompt_embedding)

    def forward(
        self,
        anchor_traj: torch.Tensor,
        pos_trajs: List[torch.Tensor],
        neg_trajs: List[torch.Tensor],
        level: str = 'low'
    ) -> tuple:
        """
        前向传播，返回 InfoNCE 损失和锚点嵌入。
        :param anchor_traj: 锚点轨迹
        :param pos_trajs: 正样本轨迹列表
        :param neg_trajs: 负样本轨迹列表
        :param level: 层次（'low', 'mid', 'high'）
        :return: (loss, anchor_embedding)
        """
        anchor_emb = self.encode_trajectory(anchor_traj, level)
        # Note: This forward pass doesn't support weighting.
        # Weighting must be handled by calling compute_contrastive_loss directly.
        loss = self.compute_contrastive_loss(anchor_traj, pos_trajs, neg_trajs, level=level)
        return loss, anchor_emb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .hpcr_prediction_head import HPCRPredictionHead

class ZeroShotHPCR(nn.Module):
    """
    Zero-shot优化的HPCR模块
    专注于在没有预训练数据的情况下快速适应新环境
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 快速适应网络
        self.rapid_adaptation_net = nn.Sequential(
            nn.Linear(config['state_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config['hidden_dim'])
        )
        
        # 元学习预测头
        self.meta_prediction_heads = nn.ModuleDict({
            'low_to_mid': HPCRPredictionHead(
                input_dim=config['hidden_dim'],
                output_dim=config['hidden_dim'] * 2,
                temperature=0.1
            ),
            'mid_to_high': HPCRPredictionHead(
                input_dim=config['hidden_dim'] * 2,
                output_dim=config['hidden_dim'] * 4,
                temperature=0.15
            )
        })
        
        # 环境特征提取器
        self.env_feature_extractor = nn.Sequential(
            nn.Linear(config['state_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 动态prompt生成器
        self.dynamic_prompt_generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, config['state_dim'])
        )
    
    def forward(self, observation: torch.Tensor, trajectory: List[Dict]) -> Dict:
        """
        Zero-shot HPCR前向传播
        """
        # 1. 快速环境适应
        env_features = self.env_feature_extractor(observation)
        adapted_representation = self.rapid_adaptation_net(observation)
        
        # 2. 动态prompt生成
        dynamic_prompt = self.dynamic_prompt_generator(env_features)
        
        # 3. 层级预测
        if len(trajectory) >= 10:  # 有足够数据时进行预测
            low_emb = self._encode_level(trajectory[-10:], 'low')
            pred_mid = self.meta_prediction_heads['low_to_mid'](low_emb)
            
            if len(trajectory) >= 50:
                mid_emb = self._encode_level(trajectory[-50:], 'mid')
                pred_high = self.meta_prediction_heads['mid_to_high'](mid_emb)
            else:
                pred_high = None
        else:
            pred_mid = None
            pred_high = None
        
        return {
            'adapted_representation': adapted_representation,
            'dynamic_prompt': dynamic_prompt,
            'pred_mid': pred_mid,
            'pred_high': pred_high,
            'env_features': env_features
        }
    
    def _encode_level(self, trajectory_segment: List[Dict], level: str) -> torch.Tensor:
        """编码轨迹段"""
        # 简化的编码逻辑
        states = torch.stack([step['state'] for step in trajectory_segment])
        return states.mean(dim=0)  # 平均池化
    
    def compute_zero_shot_loss(
        self,
        current_obs: torch.Tensor,
        trajectory: List[Dict],
        success_indicator: float
    ) -> torch.Tensor:
        """
        计算zero-shot损失
        """
        outputs = self.forward(current_obs, trajectory)
        
        # 1. 环境适应损失
        adaptation_loss = F.mse_loss(
            outputs['adapted_representation'],
            current_obs
        )
        
        # 2. 动态prompt一致性损失
        prompt_consistency_loss = F.cosine_embedding_loss(
            outputs['dynamic_prompt'],
            current_obs,
            torch.ones(current_obs.size(0), device=current_obs.device)
        )
        
        # 3. 预测损失（如果有足够数据）
        prediction_loss = torch.tensor(0.0, device=current_obs.device)
        if outputs['pred_mid'] is not None:
            # 使用当前轨迹的mid段作为目标
            if len(trajectory) >= 50:
                target_mid = self._encode_level(trajectory[-50:], 'mid')
                prediction_loss += F.mse_loss(outputs['pred_mid'], target_mid)
        
        # 4. 成功奖励损失
        success_loss = -torch.log(torch.sigmoid(success_indicator))
        
        total_loss = (
            adaptation_loss * 0.3 +
            prompt_consistency_loss * 0.3 +
            prediction_loss * 0.3 +
            success_loss * 0.1
        )
        
        return total_loss


class AdaptiveSinkhornAlign(nn.Module):
    """
    自适应Sinkhorn对齐，适用于zero-shot场景
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 动态prompt生成
        self.prompt_generator = nn.Sequential(
            nn.Linear(config['state_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, config['state_dim'])
        )
        
        # 自适应对齐网络
        self.alignment_net = nn.Sequential(
            nn.Linear(config['state_dim'] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        trajectory_embeddings: torch.Tensor,
        current_observation: torch.Tensor
    ) -> torch.Tensor:
        """
        自适应Sinkhorn对齐
        """
        # 根据当前观察生成动态prompt
        dynamic_prompt = self.prompt_generator(current_observation)
        
        # 计算对齐分数
        batch_size = trajectory_embeddings.size(0)
        prompt_expanded = dynamic_prompt.unsqueeze(0).expand(batch_size, -1)
        
        combined = torch.cat([trajectory_embeddings, prompt_expanded], dim=-1)
        alignment_scores = self.alignment_net(combined)
        
        return alignment_scores.squeeze()


class ZeroShotScoreField(nn.Module):
    """
    Zero-shot分数场，在线学习prompt embedding
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 在线prompt embedding
        self.online_prompt_embedding = nn.Parameter(
            torch.randn(config['state_dim'])
        )
        
        # 分数网络
        self.score_net = nn.Sequential(
            nn.Linear(config['state_dim'] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        trajectory_embeddings: torch.Tensor,
        success_rate: float = 0.5
    ) -> torch.Tensor:
        """
        计算分数场损失
        """
        batch_size = trajectory_embeddings.size(0)
        prompt_expanded = self.online_prompt_embedding.unsqueeze(0).expand(batch_size, -1)
        
        combined = torch.cat([trajectory_embeddings, prompt_expanded], dim=-1)
        scores = self.score_net(combined)
        
        # 根据成功率调整目标
        target_score = torch.tensor(success_rate, device=scores.device)
        score_loss = F.mse_loss(scores.squeeze(), target_score)
        
        return score_loss
    
    def update_prompt(self, success_rate: float, current_observation: torch.Tensor):
        """
        根据成功率更新prompt embedding
        """
        if success_rate > 0.7:
            # 成功时，向当前观察方向更新
            with torch.no_grad():
                self.online_prompt_embedding.data = (
                    0.9 * self.online_prompt_embedding.data +
                    0.1 * current_observation
                )
        elif success_rate < 0.3:
            # 失败时，远离当前观察
            with torch.no_grad():
                self.online_prompt_embedding.data = (
                    0.9 * self.online_prompt_embedding.data -
                    0.1 * current_observation
                ) 
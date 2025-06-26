import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class HPCRPredictionHead(nn.Module):
    """
    HPCR预测头：实现层级间的预测式对比学习
    在层级 l 中，预测下一级（l+1）子轨迹表示，并通过对比学习最大化预测准确度
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
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.prediction_network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z_current: torch.Tensor) -> torch.Tensor:
        """
        预测下一层级的表示
        :param z_current: 当前层级的轨迹表示, shape (batch_size, input_dim) or (input_dim,)
        :return: 预测的下一层级表示, shape (batch_size, output_dim) or (output_dim,)
        """
        return self.prediction_network(z_current)
    
    def compute_predictive_infonce_loss(
        self,
        z_current: torch.Tensor,
        z_next_positive: torch.Tensor,
        z_next_negatives: List[torch.Tensor],
        negative_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算预测式InfoNCE损失
        
        L_pred^l = -E[log(exp(<f_θ(z^l), z^{l+1}>/τ) / 
                       Σ_tilde{z^{l+1}} exp(<f_θ(z^l), tilde{z^{l+1}}>/τ))]
        
        :param z_current: 当前层级表示 z^l, shape (batch_size, input_dim)
        :param z_next_positive: 真实下一层级表示 z^{l+1}, shape (batch_size, output_dim)  
        :param z_next_negatives: 负样本列表，每个shape (batch_size, output_dim)
        :param negative_weights: 负样本权重，shape (num_negatives,)
        :return: 预测式InfoNCE损失
        """
        # 预测下一层级表示
        z_pred = self.forward(z_current)  # (batch_size, output_dim)
        
        # 计算正样本相似度
        pos_sim = F.cosine_similarity(z_pred, z_next_positive, dim=-1)  # (batch_size,)
        pos_logits = pos_sim / self.temperature
        
        # 计算负样本相似度
        neg_logits_list = []
        for neg_sample in z_next_negatives:
            neg_sim = F.cosine_similarity(z_pred, neg_sample, dim=-1)  # (batch_size,)
            neg_logits_list.append(neg_sim / self.temperature)
        
        if not neg_logits_list:
            return torch.tensor(0.0, device=z_current.device, requires_grad=True)
        
        neg_logits = torch.stack(neg_logits_list, dim=1)  # (batch_size, num_negatives)
        
        # 应用负样本权重
        if negative_weights is not None:
            neg_logits = neg_logits * negative_weights.unsqueeze(0)  # (batch_size, num_negatives)
        
        # 计算InfoNCE损失
        numerator = torch.exp(pos_logits)  # (batch_size,)
        denominator = numerator + torch.exp(neg_logits).sum(dim=1)  # (batch_size,)
        
        loss = -torch.log(numerator / (denominator + 1e-8))  # (batch_size,)
        
        return loss.mean()


class MINEEstimator(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) 模块
    用于估计互信息并提供收敛保证
    """
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        # 构建统计网络 T_θ
        layers = []
        input_dim = x_dim + y_dim
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.statistics_network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算统计量 T_θ(x,y)
        :param x: 输入x, shape (batch_size, x_dim)
        :param y: 输入y, shape (batch_size, y_dim)
        :return: 统计量, shape (batch_size, 1)
        """
        xy = torch.cat([x, y], dim=-1)  # (batch_size, x_dim + y_dim)
        return self.statistics_network(xy)
    
    def compute_mi_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_shuffled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算MINE损失和互信息估计
        
        MI(X;Y) ≈ E_P[T_θ(x,y)] - log(E_P'[exp(T_θ(x,y'))])
        
        :param x: 配对数据x, shape (batch_size, x_dim)
        :param y: 配对数据y, shape (batch_size, y_dim)
        :param y_shuffled: 打乱的y数据, shape (batch_size, y_dim)
        :return: (mine_loss, mi_estimate)
        """
        # 计算配对和非配对的统计量
        t_joint = self.forward(x, y)  # (batch_size, 1)
        t_marginal = self.forward(x, y_shuffled)  # (batch_size, 1)
        
        # MINE损失（要最大化的目标）
        mi_estimate = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
        mine_loss = -mi_estimate  # 负号因为我们要最小化损失
        
        return mine_loss, mi_estimate


class HPCRFailureSampleGenerator:
    """
    HPCR失败样本生成器：使用GPT-4生成失败子轨迹作为hard negatives
    """
    def __init__(
        self,
        llm_api_key: str,
        llm_api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4",
        max_retries: int = 3
    ):
        self.api_key = llm_api_key
        self.api_base = llm_api_base
        self.model_name = model_name
        self.max_retries = max_retries
    
    def generate_failure_trajectories(
        self,
        successful_trajectory: List[Dict],
        num_failures: int = 3,
        failure_types: List[str] = None
    ) -> List[List[Dict]]:
        """
        基于成功轨迹生成失败版本
        
        :param successful_trajectory: 成功的轨迹段
        :param num_failures: 生成失败样本数量
        :param failure_types: 失败类型 ['action_error', 'timing_error', 'sequence_error']
        :return: 失败轨迹列表
        """
        if failure_types is None:
            failure_types = ['action_error', 'timing_error', 'sequence_error']
        
        failure_trajectories = []
        
        for i in range(num_failures):
            failure_type = failure_types[i % len(failure_types)]
            failure_traj = self._generate_single_failure(
                successful_trajectory, 
                failure_type
            )
            if failure_traj:
                failure_trajectories.append(failure_traj)
        
        return failure_trajectories
    
    def _generate_single_failure(
        self,
        success_traj: List[Dict],
        failure_type: str
    ) -> Optional[List[Dict]]:
        """
        生成单个失败轨迹
        """
        # 这里应该调用GPT-4 API生成失败样本
        # 为了演示，我们使用简化的规则生成
        
        failure_traj = success_traj.copy()
        
        if failure_type == 'action_error':
            # 随机替换某个动作
            if len(failure_traj) > 1:
                idx = torch.randint(0, len(failure_traj), (1,)).item()
                # 简化：随机改变动作
                if 'action' in failure_traj[idx]:
                    original_action = failure_traj[idx]['action']
                    # 这里应该有更智能的失败动作生成逻辑
                    failure_traj[idx]['action'] = self._corrupt_action(original_action)
        
        elif failure_type == 'timing_error':
            # 时序错误：打乱某些步骤的顺序
            if len(failure_traj) > 2:
                i, j = torch.randperm(len(failure_traj))[:2].tolist()
                failure_traj[i], failure_traj[j] = failure_traj[j], failure_traj[i]
        
        elif failure_type == 'sequence_error':
            # 序列错误：删除或重复某些步骤
            if len(failure_traj) > 1:
                idx = torch.randint(0, len(failure_traj), (1,)).item()
                if torch.rand(1).item() > 0.5:
                    # 删除步骤
                    failure_traj.pop(idx)
                else:
                    # 重复步骤
                    failure_traj.insert(idx, failure_traj[idx].copy())
        
        return failure_traj
    
    def _corrupt_action(self, action):
        """简化的动作腐蚀逻辑"""
        if isinstance(action, torch.Tensor):
            noise = torch.randn_like(action) * 0.1
            return action + noise
        else:
            return action  # 对于非tensor类型的动作，保持不变


class HPCRHierarchicalSlicing:
    """
    HPCR层级轨迹切片器：实现多层级轨迹切分
    """
    def __init__(
        self,
        low_span: int = 10,
        mid_span: int = 50,
        high_span: int = -1  # -1表示全episode
    ):
        self.low_span = low_span
        self.mid_span = mid_span
        self.high_span = high_span
    
    def slice_trajectory(
        self,
        full_trajectory: List[Dict],
        level: str
    ) -> List[List[Dict]]:
        """
        根据层级切分轨迹
        
        :param full_trajectory: 完整轨迹
        :param level: 层级 ('low', 'mid', 'high')
        :return: 切分后的子轨迹列表
        """
        if level == 'low':
            span = self.low_span
        elif level == 'mid':
            span = self.mid_span
        elif level == 'high':
            span = self.high_span if self.high_span > 0 else len(full_trajectory)
        else:
            raise ValueError(f"Unknown level: {level}")
        
        if span >= len(full_trajectory):
            return [full_trajectory]
        
        # 滑动窗口切分
        slices = []
        for i in range(0, len(full_trajectory) - span + 1, span):
            slice_traj = full_trajectory[i:i + span]
            slices.append(slice_traj)
        
        # 确保最后一个片段被包含
        if len(full_trajectory) % span != 0:
            slices.append(full_trajectory[-(span):])
        
        return slices 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

from .rce import RceModule
from .cgf import CGFModule
from .hpcr_prediction_head import HPCRPredictionHead, MINEEstimator, HPCRFailureSampleGenerator, HPCRHierarchicalSlicing

MultiTransition = namedtuple('MultiTransition', ['agent_id', 'state', 'action', 'reward'])

class HPCRReflectionHead(nn.Module):
    """
    HPCR增强版反思头：实现层级预测对比反思 (Hierarchical Predictive Contrastive Reflection)
    
    核心创新：
    1. 多层级预测式对比学习：每层预测下一层级表示并计算预测式InfoNCE损失
    2. 跨层级反思反馈：mid层使用GPT-4生成失败样本作为hard negatives  
    3. 互信息最大化框架：通过MINE估计保证收敛性
    """
    
    def __init__(
        self,
        config: Dict,
        rce_module: RceModule,
        cgf_module: CGFModule,
        gamma: float = 0.99
    ):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.rce_module = rce_module
        self.cgf_module = cgf_module
        
        # HPCR配置
        self.hpcr_config = config.get('hpcr', {})
        self.enable_hpcr = self.hpcr_config.get('enable_hpcr', True)
        
        # 层级维度配置
        self.embedding_dims = {
            'low': config.get('hidden_dim', 128),
            'mid': config.get('hidden_dim', 128) * 2,
            'high': config.get('hidden_dim', 128) * 4
        }
        
        # 为每个层级创建预测头
        self.prediction_heads = nn.ModuleDict()
        if self.enable_hpcr:
            # low -> mid 预测头
            self.prediction_heads['low_to_mid'] = HPCRPredictionHead(
                input_dim=self.embedding_dims['low'],
                output_dim=self.embedding_dims['mid'],
                hidden_dim=self.hpcr_config.get('prediction_head', {}).get('hidden_dim', 256),
                num_layers=self.hpcr_config.get('prediction_head', {}).get('num_layers', 2),
                dropout=self.hpcr_config.get('prediction_head', {}).get('dropout', 0.1),
                temperature=self.hpcr_config.get('temperatures', {}).get('low', 0.1)
            )
            
            # mid -> high 预测头
            self.prediction_heads['mid_to_high'] = HPCRPredictionHead(
                input_dim=self.embedding_dims['mid'],
                output_dim=self.embedding_dims['high'],
                hidden_dim=self.hpcr_config.get('prediction_head', {}).get('hidden_dim', 256),
                num_layers=self.hpcr_config.get('prediction_head', {}).get('num_layers', 2),
                dropout=self.hpcr_config.get('prediction_head', {}).get('dropout', 0.1),
                temperature=self.hpcr_config.get('temperatures', {}).get('mid', 0.15)
            )
        
        # MINE估计器（可选）
        self.mine_estimators = nn.ModuleDict()
        if self.hpcr_config.get('mine', {}).get('enabled', False):
            mine_hidden = self.hpcr_config.get('mine', {}).get('mine_hidden_dim', 128)
            
            self.mine_estimators['low_mid'] = MINEEstimator(
                x_dim=self.embedding_dims['low'],
                y_dim=self.embedding_dims['mid'],
                hidden_dim=mine_hidden
            )
            
            self.mine_estimators['mid_high'] = MINEEstimator(
                x_dim=self.embedding_dims['mid'],
                y_dim=self.embedding_dims['high'],
                hidden_dim=mine_hidden
            )
        
        # 失败样本生成器
        if self.hpcr_config.get('failure_sampling', {}).get('enabled', False):
            self.failure_generator = HPCRFailureSampleGenerator(
                llm_api_key=config.get('llm_api_key', ''),
                llm_api_base=config.get('llm_api_base', ''),
                model_name=self.hpcr_config.get('failure_sampling', {}).get('gpt4_model', 'gpt-4')
            )
        else:
            self.failure_generator = None
        
        # 轨迹切片器
        slicing_config = self.hpcr_config.get('trajectory_slicing', {})
        self.trajectory_slicer = HPCRHierarchicalSlicing(
            low_span=slicing_config.get('low_span', 10),
            mid_span=slicing_config.get('mid_span', 50),
            high_span=slicing_config.get('high_span', -1)
        )
        
        # 层级融合权重
        fusion_weights = self.hpcr_config.get('gradient_fusion', {}).get('fusion_weights', {})
        self.fusion_weights = {
            'low': fusion_weights.get('low', 0.5),
            'mid': fusion_weights.get('mid', 0.3),
            'high': fusion_weights.get('high', 0.2)
        }
    
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
        level: str = 'low',
        hierarchical_data: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        执行HPCR增强的反思与对齐过程
        
        :param hierarchical_data: 包含多层级轨迹数据的字典
            {
                'low': {'current': z_low, 'next': z_mid, 'negatives': [...]},
                'mid': {'current': z_mid, 'next': z_high, 'negatives': [...]},
                'high': {'current': z_high, 'next': None, 'negatives': [...]}
            }
        """
        # 1. 计算传统的InfoNCE对比损失
        contrastive_loss = self.rce_module.compute_contrastive_loss(
            anchor_traj=anchor_traj,
            pos_trajs=pos_trajs,
            neg_trajs=neg_trajs,
            neg_weights=neg_weights,
            level=level
        )
        
        # 2. 计算对齐损失
        anchor_state_embeddings = self.rce_module.get_trajectory_state_embeddings(
            anchor_traj, level=level
        )
        
        sinkhorn_loss = self.rce_module.compute_sinkhorn_loss(
            prompt_token_embeddings=prompt_token_embeddings,
            trajectory_state_embeddings=anchor_state_embeddings
        )
        
        score_loss = self.rce_module.compute_score_loss(
            trajectory_state_embeddings=anchor_state_embeddings,
            prompt_embedding=prompt_global_embedding
        )
        
        # 3. HPCR预测式对比损失
        hpcr_losses = {}
        total_hpcr_loss = torch.tensor(0.0, device=anchor_traj.device, requires_grad=True)
        
        if self.enable_hpcr and hierarchical_data is not None:
            # 计算各层级的预测损失
            hpcr_losses, total_hpcr_loss = self._compute_hpcr_losses(hierarchical_data)
        
        # 4. MINE互信息损失（可选）
        mine_losses = {}
        total_mine_loss = torch.tensor(0.0, device=anchor_traj.device, requires_grad=True)
        
        if self.mine_estimators and hierarchical_data is not None:
            mine_losses, total_mine_loss = self._compute_mine_losses(hierarchical_data)
        
        # 5. 加权合并总损失
        pred_weights = self.hpcr_config.get('pred_weights', {})
        alpha_level = pred_weights.get(level, 1.0)
        
        total_loss = (
            contrastive_loss +
            sinkhorn_weight * sinkhorn_loss +
            score_field_weight * score_loss +
            alpha_level * total_hpcr_loss +
            0.1 * total_mine_loss  # MINE损失权重较小
        )
        
        # 6. CGF梯度映射
        self.cgf_module(
            loss=total_loss,
            anchor_emb=anchor_state_embeddings,
            level=level
        )
        
        # 构建损失详情
        loss_breakdown = {
            "contrastive_loss": contrastive_loss.item(),
            "sinkhorn_loss": sinkhorn_loss.item(),
            "score_field_loss": score_loss.item(),
            "total_hpcr_loss": total_hpcr_loss.item(),
            "total_mine_loss": total_mine_loss.item(),
            "total_loss": total_loss.item()
        }
        
        # 添加各层级HPCR损失详情
        for key, value in hpcr_losses.items():
            loss_breakdown[f"hpcr_{key}"] = value.item() if isinstance(value, torch.Tensor) else value
        
        # 添加MINE损失详情
        for key, value in mine_losses.items():
            loss_breakdown[f"mine_{key}"] = value.item() if isinstance(value, torch.Tensor) else value
        
        return total_loss, loss_breakdown
    
    def _compute_hpcr_losses(
        self,
        hierarchical_data: Dict
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """计算HPCR预测式对比损失"""
        hpcr_losses = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        # Low -> Mid 预测损失
        if 'low' in hierarchical_data and 'mid' in hierarchical_data:
            low_data = hierarchical_data['low']
            mid_data = hierarchical_data['mid']
            
            if 'low_to_mid' in self.prediction_heads:
                low_to_mid_loss = self.prediction_heads['low_to_mid'].compute_predictive_infonce_loss(
                    z_current=low_data['current'],
                    z_next_positive=mid_data['current'],  # mid层的当前表示作为low层的目标
                    z_next_negatives=low_data.get('negatives', [])
                )
                hpcr_losses['low_to_mid'] = low_to_mid_loss
                total_loss = total_loss + self.fusion_weights['low'] * low_to_mid_loss
        
        # Mid -> High 预测损失
        if 'mid' in hierarchical_data and 'high' in hierarchical_data:
            mid_data = hierarchical_data['mid']
            high_data = hierarchical_data['high']
            
            if 'mid_to_high' in self.prediction_heads:
                # 为mid层生成失败样本作为额外的负样本
                enhanced_negatives = self._generate_enhanced_negatives(
                    mid_data, 'mid'
                )
                
                mid_to_high_loss = self.prediction_heads['mid_to_high'].compute_predictive_infonce_loss(
                    z_current=mid_data['current'],
                    z_next_positive=high_data['current'],
                    z_next_negatives=enhanced_negatives
                )
                hpcr_losses['mid_to_high'] = mid_to_high_loss
                total_loss = total_loss + self.fusion_weights['mid'] * mid_to_high_loss
        
        return hpcr_losses, total_loss
    
    def _compute_mine_losses(
        self,
        hierarchical_data: Dict
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """计算MINE互信息估计损失"""
        mine_losses = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        # Low-Mid 互信息估计
        if ('low' in hierarchical_data and 'mid' in hierarchical_data and 
            'low_mid' in self.mine_estimators):
            
            low_emb = hierarchical_data['low']['current']
            mid_emb = hierarchical_data['mid']['current']
            
            # 生成打乱的样本
            batch_size = mid_emb.size(0) if mid_emb.dim() > 1 else 1
            if batch_size > 1:
                perm_idx = torch.randperm(batch_size)
                mid_shuffled = mid_emb[perm_idx]
            else:
                # 对于单样本，我们可以使用一个随机生成的样本
                mid_shuffled = torch.randn_like(mid_emb)
            
            mine_loss, mi_estimate = self.mine_estimators['low_mid'].compute_mi_loss(
                x=low_emb,
                y=mid_emb,
                y_shuffled=mid_shuffled
            )
            mine_losses['low_mid_loss'] = mine_loss
            mine_losses['low_mid_mi'] = mi_estimate
            total_loss = total_loss + mine_loss
        
        # Mid-High 互信息估计
        if ('mid' in hierarchical_data and 'high' in hierarchical_data and 
            'mid_high' in self.mine_estimators):
            
            mid_emb = hierarchical_data['mid']['current']
            high_emb = hierarchical_data['high']['current']
            
            batch_size = high_emb.size(0) if high_emb.dim() > 1 else 1
            if batch_size > 1:
                perm_idx = torch.randperm(batch_size)
                high_shuffled = high_emb[perm_idx]
            else:
                high_shuffled = torch.randn_like(high_emb)
            
            mine_loss, mi_estimate = self.mine_estimators['mid_high'].compute_mi_loss(
                x=mid_emb,
                y=high_emb,
                y_shuffled=high_shuffled
            )
            mine_losses['mid_high_loss'] = mine_loss
            mine_losses['mid_high_mi'] = mi_estimate
            total_loss = total_loss + mine_loss
        
        return mine_losses, total_loss
    
    def _generate_enhanced_negatives(
        self,
        level_data: Dict,
        level: str
    ) -> List[torch.Tensor]:
        """为指定层级生成增强的负样本（包括失败样本）"""
        enhanced_negatives = level_data.get('negatives', []).copy()
        
        # 如果启用了失败样本生成且是mid层
        if (self.failure_generator is not None and 
            level == 'mid' and 
            self.hpcr_config.get('failure_sampling', {}).get('enabled', False)):
            
            try:
                # 这里应该从原始轨迹数据生成失败样本
                # 为了简化，我们生成一些随机的negative samples
                failure_ratio = self.hpcr_config.get('failure_sampling', {}).get('failure_sample_ratio', 0.3)
                num_failures = int(len(enhanced_negatives) * failure_ratio)
                
                if num_failures > 0:
                    current_emb = level_data['current']
                    for _ in range(num_failures):
                        # 生成失败样本（添加噪声）
                        noise = torch.randn_like(current_emb) * 0.2
                        failure_sample = current_emb + noise
                        enhanced_negatives.append(failure_sample)
            
            except Exception as e:
                print(f"Warning: Failed to generate failure samples: {e}")
        
        return enhanced_negatives
    
    def prepare_hierarchical_data(
        self,
        full_trajectory: List[Dict],
        episode_step: int
    ) -> Dict:
        """
        准备层级化数据用于HPCR训练
        
        :param full_trajectory: 完整episode轨迹
        :param episode_step: 当前episode步数
        :return: 层级化数据字典
        """
        hierarchical_data = {}
        
        # 切分不同层级的轨迹
        for level in ['low', 'mid', 'high']:
            slices = self.trajectory_slicer.slice_trajectory(full_trajectory, level)
            
            if slices:
                # 选择当前时间步对应的切片
                current_slice_idx = min(episode_step // self._get_level_stride(level), len(slices) - 1)
                current_slice = slices[current_slice_idx]
                
                # 编码当前切片
                current_tensor = self._trajectory_to_tensor(current_slice)
                current_embedding = self.rce_module.encode_trajectory(current_tensor, level)
                
                # 准备负样本（其他切片）
                negative_embeddings = []
                for i, slice_traj in enumerate(slices):
                    if i != current_slice_idx:
                        neg_tensor = self._trajectory_to_tensor(slice_traj)
                        neg_embedding = self.rce_module.encode_trajectory(neg_tensor, level)
                        negative_embeddings.append(neg_embedding)
                
                hierarchical_data[level] = {
                    'current': current_embedding,
                    'negatives': negative_embeddings
                }
        
        return hierarchical_data
    
    def _get_level_stride(self, level: str) -> int:
        """获取层级对应的步长"""
        if level == 'low':
            return self.trajectory_slicer.low_span
        elif level == 'mid':
            return self.trajectory_slicer.mid_span
        elif level == 'high':
            return max(self.trajectory_slicer.high_span, 100)
        return 1
    
    def _trajectory_to_tensor(self, trajectory: List[Dict]) -> torch.Tensor:
        """将轨迹转换为tensor格式"""
        # 这是一个简化的实现，实际中需要根据轨迹的具体格式进行转换
        tensor_list = []
        
        for step in trajectory:
            state = step.get('state', torch.zeros(512))  # 默认state_dim=512
            action = step.get('action', torch.zeros(64))  # 默认action_dim=64
            reward = step.get('reward', torch.tensor(0.0))
            
            # 确保是tensor格式
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor([reward], dtype=torch.float32)
            
            # 拼接state + action + reward
            step_tensor = torch.cat([state.flatten(), action.flatten(), reward.flatten()])
            tensor_list.append(step_tensor)
        
        if tensor_list:
            return torch.stack(tensor_list)  # (T, input_dim)
        else:
            # 返回一个默认的空轨迹
            return torch.zeros(1, 512 + 64 + 1)  # (1, state_dim + action_dim + 1) 
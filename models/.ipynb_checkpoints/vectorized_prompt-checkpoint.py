#!/usr/bin/env python3
"""
向量化提示模块 - 将state_dim、action_dim等信息序列化为prompt metadata
支持transition modeling等任务的上下文增强
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict


@dataclass
class VectorMetadata:
    """向量化元数据"""
    state_dim: int
    action_dim: int
    state_dtype: str = "float32"
    action_dtype: str = "float32"
    state_range: Optional[Dict[str, float]] = None  # {"min": 0.0, "max": 1.0}
    action_range: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None


@dataclass
class TaskContext:
    """任务上下文信息"""
    task_type: str  # "transition_modeling", "action_sequencing", etc.
    environment: str  # "virtualhome", "behavior", etc.
    agent_count: int = 1
    episode_length: Optional[int] = None
    reward_range: Optional[Dict[str, float]] = None


class VectorizedPromptGenerator:
    """向量化提示生成器"""
    
    def __init__(self, metadata: VectorMetadata, context: TaskContext):
        self.metadata = metadata
        self.context = context
        
    def serialize_vector_metadata(self) -> str:
        """将向量元数据序列化为可读的prompt部分"""
        metadata_str = f"""
### VECTOR METADATA ###
State Space Configuration:
- Dimensions: {self.metadata.state_dim}
- Data Type: {self.metadata.state_dtype}
- Range: {self.metadata.state_range or "[-inf, +inf]"}
{f"- Feature Names: {self.metadata.feature_names}" if self.metadata.feature_names else ""}

Action Space Configuration:
- Dimensions: {self.metadata.action_dim}
- Data Type: {self.metadata.action_dtype}
- Range: {self.metadata.action_range or "[-inf, +inf]"}
{f"- Action Names: {self.metadata.action_names}" if self.metadata.action_names else ""}

Task Context:
- Type: {self.context.task_type}
- Environment: {self.context.environment}
- Agent Count: {self.context.agent_count}
{f"- Episode Length: {self.context.episode_length}" if self.context.episode_length else ""}
{f"- Reward Range: {self.context.reward_range}" if self.context.reward_range else ""}
### END METADATA ###
"""
        return metadata_str.strip()
    
    def vectorize_state(self, state: Union[torch.Tensor, np.ndarray, List]) -> str:
        """将状态向量序列化为文本"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        elif isinstance(state, list):
            state = np.array(state)
            
        # 格式化向量为可读形式
        if len(state.shape) == 1:
            # 单个状态向量
            formatted = ", ".join([f"{val:.4f}" for val in state])
            if self.metadata.feature_names and len(self.metadata.feature_names) == len(state):
                # 如果有特征名称，创建命名向量
                named_features = [f"{name}:{val:.4f}" for name, val in zip(self.metadata.feature_names, state)]
                return f"State Vector: [{', '.join(named_features)}]"
            else:
                return f"State Vector: [{formatted}]"
        else:
            # 批量状态向量
            batch_size = state.shape[0]
            formatted_batch = []
            for i, s in enumerate(state):
                formatted = ", ".join([f"{val:.4f}" for val in s])
                formatted_batch.append(f"  State[{i}]: [{formatted}]")
            return f"State Batch (size={batch_size}):\n" + "\n".join(formatted_batch)
    
    def vectorize_action(self, action: Union[torch.Tensor, np.ndarray, List]) -> str:
        """将动作向量序列化为文本"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)
            
        # 格式化动作向量
        if len(action.shape) == 1:
            # 单个动作向量
            formatted = ", ".join([f"{val:.4f}" for val in action])
            if self.metadata.action_names and len(self.metadata.action_names) == len(action):
                # 如果有动作名称，创建命名向量
                named_actions = [f"{name}:{val:.4f}" for name, val in zip(self.metadata.action_names, action)]
                return f"Action Vector: [{', '.join(named_actions)}]"
            else:
                return f"Action Vector: [{formatted}]"
        else:
            # 批量动作向量
            batch_size = action.shape[0]
            formatted_batch = []
            for i, a in enumerate(action):
                formatted = ", ".join([f"{val:.4f}" for val in a])
                formatted_batch.append(f"  Action[{i}]: [{formatted}]")
            return f"Action Batch (size={batch_size}):\n" + "\n".join(formatted_batch)
    
    def create_transition_prompt(
        self, 
        current_state: Union[torch.Tensor, np.ndarray, List],
        action: Union[torch.Tensor, np.ndarray, List],
        additional_context: str = ""
    ) -> str:
        """创建transition modeling的完整提示"""
        
        metadata_section = self.serialize_vector_metadata()
        current_state_str = self.vectorize_state(current_state)
        action_str = self.vectorize_action(action)
        
        prompt = f"""
{metadata_section}

### TRANSITION MODELING TASK ###

Given the current state vector and the action vector being executed, predict the resulting next state vector after the action is performed.

Current State:
{current_state_str}

Action Being Executed:
{action_str}

{additional_context}

IMPORTANT: Your response should be a next state vector in the same format and dimensionality as the current state. 
Output the predicted next state as a JSON array of {self.metadata.state_dim} float values.

Example format: [0.1234, 0.5678, 0.9012, ...]

Predicted Next State:"""
        
        return prompt.strip()
    
    def create_trajectory_prompt(
        self,
        state_sequence: Union[torch.Tensor, np.ndarray, List],
        action_sequence: Union[torch.Tensor, np.ndarray, List],
        additional_context: str = ""
    ) -> str:
        """创建轨迹级别的提示"""
        
        metadata_section = self.serialize_vector_metadata()
        
        # 处理序列数据
        if isinstance(state_sequence, torch.Tensor):
            state_sequence = state_sequence.cpu().numpy()
        if isinstance(action_sequence, torch.Tensor):
            action_sequence = action_sequence.cpu().numpy()
            
        trajectory_str = ""
        seq_len = len(state_sequence)
        
        for t in range(seq_len):
            state_str = self.vectorize_state(state_sequence[t])
            if t < len(action_sequence):
                action_str = self.vectorize_action(action_sequence[t])
                trajectory_str += f"\nStep {t}:\n  {state_str}\n  {action_str}\n"
            else:
                trajectory_str += f"\nStep {t} (Final):\n  {state_str}\n"
        
        prompt = f"""
{metadata_section}

### TRAJECTORY ANALYSIS TASK ###

Analyze the following state-action trajectory and provide insights or predictions based on the sequence.

Trajectory:
{trajectory_str}

{additional_context}

Please analyze this trajectory and provide your response:"""
        
        return prompt.strip()
    
    def create_policy_prompt(
        self,
        current_state: Union[torch.Tensor, np.ndarray, List],
        goal_state: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        additional_context: str = ""
    ) -> str:
        """创建策略生成的提示"""
        
        metadata_section = self.serialize_vector_metadata()
        current_state_str = self.vectorize_state(current_state)
        
        goal_section = ""
        if goal_state is not None:
            goal_state_str = self.vectorize_state(goal_state)
            goal_section = f"\nGoal State:\n{goal_state_str}\n"
        
        prompt = f"""
{metadata_section}

### POLICY GENERATION TASK ###

Given the current state{" and goal state" if goal_state is not None else ""}, generate an appropriate action vector.

Current State:
{current_state_str}
{goal_section}
{additional_context}

IMPORTANT: Your response should be an action vector with {self.metadata.action_dim} dimensions.
Output the recommended action as a JSON array of {self.metadata.action_dim} float values.

Example format: [0.1234, 0.5678, 0.9012, ...]

Recommended Action:"""
        
        return prompt.strip()


class PromptEnhancer:
    """提示增强器 - 为现有提示添加向量化信息"""
    
    @staticmethod
    def enhance_with_metadata(
        original_prompt: str,
        metadata: VectorMetadata,
        context: TaskContext
    ) -> str:
        """为现有提示添加元数据增强"""
        generator = VectorizedPromptGenerator(metadata, context)
        metadata_section = generator.serialize_vector_metadata()
        
        enhanced_prompt = f"""
{metadata_section}

### ORIGINAL TASK ###
{original_prompt}

### ENHANCED INSTRUCTIONS ###
Please consider the vector metadata above when processing this task. The dimensional information provides context about the state and action spaces you're working with.
"""
        return enhanced_prompt.strip()
    
    @staticmethod
    def inject_vector_context(
        prompt: str,
        vectors: Dict[str, Union[torch.Tensor, np.ndarray, List]],
        metadata: VectorMetadata
    ) -> str:
        """将向量信息注入到现有提示中"""
        generator = VectorizedPromptGenerator(metadata, TaskContext("general", "unknown"))
        
        vector_sections = []
        for name, vector in vectors.items():
            if "state" in name.lower():
                vector_str = generator.vectorize_state(vector)
            elif "action" in name.lower():
                vector_str = generator.vectorize_action(vector)
            else:
                # 通用向量处理
                if isinstance(vector, torch.Tensor):
                    vector = vector.cpu().numpy()
                elif isinstance(vector, list):
                    vector = np.array(vector)
                formatted = ", ".join([f"{val:.4f}" for val in vector.flatten()])
                vector_str = f"{name}: [{formatted}]"
            
            vector_sections.append(vector_str)
        
        vector_context = "\n".join(vector_sections)
        
        enhanced_prompt = f"""
### VECTOR CONTEXT ###
{vector_context}

### TASK ###
{prompt}

Please use the vector context above to inform your response.
"""
        return enhanced_prompt.strip()


# 使用示例和工厂函数
def create_virtualhome_metadata(state_dim: int = 64, action_dim: int = 8) -> VectorMetadata:
    """创建VirtualHome环境的元数据"""
    return VectorMetadata(
        state_dim=state_dim,
        action_dim=action_dim,
        state_dtype="float32",
        action_dtype="float32",
        state_range={"min": -1.0, "max": 1.0},
        action_range={"min": -1.0, "max": 1.0},
        feature_names=[f"state_feature_{i}" for i in range(state_dim)],
        action_names=[f"action_dim_{i}" for i in range(action_dim)]
    )


def create_transition_context() -> TaskContext:
    """创建transition modeling的任务上下文"""
    return TaskContext(
        task_type="transition_modeling",
        environment="virtualhome",
        agent_count=1,
        episode_length=100,
        reward_range={"min": -1.0, "max": 1.0}
    ) 
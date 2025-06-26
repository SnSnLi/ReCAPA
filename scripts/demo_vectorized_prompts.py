#!/usr/bin/env python3
"""
向量化提示模块演示脚本
展示如何使用VectorizedPromptGenerator和LLMTrajectory的增强功能
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.vectorized_prompt import (
    VectorizedPromptGenerator, 
    VectorMetadata, 
    TaskContext, 
    PromptEnhancer,
    create_virtualhome_metadata,
    create_transition_context
)
from models.llm_trajectory import LLMTrajectory


def demo_basic_vectorized_prompts():
    """演示基本的向量化提示功能"""
    print("=== 基本向量化提示演示 ===\n")
    
    # 创建元数据和上下文
    metadata = VectorMetadata(
        state_dim=4,
        action_dim=2,
        state_dtype="float32",
        action_dtype="float32",
        state_range={"min": -1.0, "max": 1.0},
        action_range={"min": -1.0, "max": 1.0},
        feature_names=["position_x", "position_y", "velocity_x", "velocity_y"],
        action_names=["acceleration_x", "acceleration_y"]
    )
    
    context = TaskContext(
        task_type="transition_modeling",
        environment="virtualhome", 
        agent_count=1,
        episode_length=100,
        reward_range={"min": -1.0, "max": 1.0}
    )
    
    generator = VectorizedPromptGenerator(metadata, context)
    
    # 演示元数据序列化
    print("1. 序列化的向量元数据:")
    print(generator.serialize_vector_metadata())
    print("\n" + "="*50 + "\n")
    
    # 演示状态向量化
    current_state = [0.5, -0.3, 0.1, 0.2]
    print("2. 状态向量序列化:")
    print(generator.vectorize_state(current_state))
    print("\n" + "="*50 + "\n")
    
    # 演示动作向量化
    action = [0.8, -0.5]
    print("3. 动作向量序列化:")
    print(generator.vectorize_action(action))
    print("\n" + "="*50 + "\n")
    
    # 演示完整的transition提示
    print("4. 完整的transition modeling提示:")
    transition_prompt = generator.create_transition_prompt(
        current_state, 
        action,
        "The agent is navigating in a 2D environment with physics simulation."
    )
    print(transition_prompt)
    print("\n" + "="*50 + "\n")


def demo_batch_processing():
    """演示批量处理功能"""
    print("=== 批量处理演示 ===\n")
    
    metadata = create_virtualhome_metadata(state_dim=6, action_dim=3)
    context = create_transition_context()
    generator = VectorizedPromptGenerator(metadata, context)
    
    # 批量状态和动作
    batch_states = np.random.randn(3, 6)  # 3个状态，每个6维
    batch_actions = np.random.randn(3, 3)  # 3个动作，每个3维
    
    print("1. 批量状态处理:")
    print(generator.vectorize_state(batch_states))
    print("\n" + "="*30 + "\n")
    
    print("2. 批量动作处理:")
    print(generator.vectorize_action(batch_actions))
    print("\n" + "="*50 + "\n")


def demo_trajectory_prompts():
    """演示轨迹级别的提示"""
    print("=== 轨迹级别提示演示 ===\n")
    
    metadata = VectorMetadata(
        state_dim=3,
        action_dim=2,
        feature_names=["x", "y", "orientation"],
        action_names=["linear_vel", "angular_vel"]
    )
    context = TaskContext("trajectory_analysis", "virtualhome")
    generator = VectorizedPromptGenerator(metadata, context)
    
    # 创建示例轨迹
    trajectory_states = [
        [0.0, 0.0, 0.0],    # 起始位置
        [0.1, 0.0, 0.1],    # 轻微移动
        [0.2, 0.0, 0.2],    # 继续移动
        [0.3, 0.1, 0.3]     # 最终位置
    ]
    
    trajectory_actions = [
        [0.1, 0.1],  # 前进+转弯
        [0.1, 0.1],  # 继续
        [0.1, 0.1]   # 最后一步
    ]
    
    print("轨迹分析提示:")
    trajectory_prompt = generator.create_trajectory_prompt(
        trajectory_states,
        trajectory_actions,
        "Analyze this robot navigation trajectory for efficiency and safety."
    )
    print(trajectory_prompt)
    print("\n" + "="*50 + "\n")


def demo_prompt_enhancement():
    """演示提示增强功能"""
    print("=== 提示增强演示 ===\n")
    
    # 原始提示
    original_prompt = """
    Given the current robot state and the action 'move forward', 
    predict what the next state will be.
    """
    
    metadata = create_virtualhome_metadata(state_dim=8, action_dim=4)
    context = create_transition_context()
    
    print("1. 原始提示:")
    print(original_prompt.strip())
    print("\n" + "="*30 + "\n")
    
    print("2. 增强后的提示:")
    enhanced_prompt = PromptEnhancer.enhance_with_metadata(
        original_prompt,
        metadata,
        context
    )
    print(enhanced_prompt)
    print("\n" + "="*50 + "\n")


def demo_vector_injection():
    """演示向量注入功能"""
    print("=== 向量注入演示 ===\n")
    
    original_prompt = "Predict the next action based on the current situation."
    
    # 创建一些示例向量
    vectors = {
        "current_state": [0.1, 0.2, 0.3, 0.4],
        "previous_action": [0.8, 0.2],
        "goal_vector": [1.0, 1.0, 0.0, 0.0]
    }
    
    metadata = VectorMetadata(state_dim=4, action_dim=2)
    
    print("1. 原始提示:")
    print(original_prompt)
    print("\n" + "="*30 + "\n")
    
    print("2. 注入向量后的提示:")
    injected_prompt = PromptEnhancer.inject_vector_context(
        original_prompt,
        vectors,
        metadata
    )
    print(injected_prompt)
    print("\n" + "="*50 + "\n")


def demo_llm_integration():
    """演示与LLMTrajectory的集成（不实际调用API）"""
    print("=== LLMTrajectory集成演示 ===\n")
    
    # 创建LLMTrajectory实例（不提供API key，仅演示接口）
    llm = LLMTrajectory(
        state_dim=6,
        action_dim=3,
        enable_vectorized_prompts=True
    )
    
    print("1. 默认向量元数据:")
    print(f"State dim: {llm.vector_metadata.state_dim}")
    print(f"Action dim: {llm.vector_metadata.action_dim}")
    print(f"State dtype: {llm.vector_metadata.state_dtype}")
    print("\n" + "="*30 + "\n")
    
    # 更新元数据
    print("2. 更新向量元数据:")
    llm.update_vector_metadata(
        feature_names=["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"],
        action_names=["force_x", "force_y", "force_z"],
        state_range={"min": -10.0, "max": 10.0}
    )
    print(f"Feature names: {llm.vector_metadata.feature_names}")
    print(f"State range: {llm.vector_metadata.state_range}")
    print("\n" + "="*30 + "\n")
    
    # 演示提示生成（不调用API）
    print("3. 生成向量化提示示例:")
    current_state = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    action = [0.5, -0.3, 0.1]
    
    if llm.enable_vectorized_prompts:
        prompt = llm.prompt_generator.create_transition_prompt(
            current_state,
            action,
            "Robot is performing object manipulation task."
        )
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n" + "="*50 + "\n")


def main():
    """主演示函数"""
    print("向量化提示模块功能演示")
    print("=" * 60)
    print()
    
    # 运行各个演示
    demo_basic_vectorized_prompts()
    demo_batch_processing()
    demo_trajectory_prompts()
    demo_prompt_enhancement()
    demo_vector_injection()
    demo_llm_integration()
    
    print("演示完成！")
    print("\n使用说明:")
    print("1. VectorizedPromptGenerator: 核心向量化提示生成器")
    print("2. PromptEnhancer: 用于增强现有提示")
    print("3. LLMTrajectory增强版: 集成向量化提示支持")
    print("4. 支持torch.Tensor, numpy.ndarray, 和list等多种输入格式")
    print("5. 自动处理批量数据和序列数据")


if __name__ == "__main__":
    main() 
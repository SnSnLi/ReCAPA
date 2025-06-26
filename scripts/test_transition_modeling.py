#!/usr/bin/env python3
"""
测试Transition Modeling任务的脚本
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.llm_trajectory import LLMTrajectory

def load_transition_prompts(prompt_file):
    """加载transition modeling的提示"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def test_transition_modeling(num_samples=5):
    """测试transition modeling"""
    
    # 加载提示数据
    prompt_file = "./temp_prompts/virtualhome/generate_prompts/transition_modeling/helm_prompt.json"
    if not os.path.exists(prompt_file):
        print(f"错误: 找不到提示文件 {prompt_file}")
        return
    
    prompts = load_transition_prompts(prompt_file)
    print(f"加载了 {len(prompts)} 个transition modeling任务")
    
    # 初始化LLM
    llm = LLMTrajectory(
        model_name="deepseek-chat",
        api_base="https://api.deepseek.com/v1"
    )
    
    # 测试前几个样本
    results = []
    test_samples = prompts[:num_samples]
    
    for i, sample in enumerate(test_samples):
        identifier = sample.get('identifier', f'sample_{i}')
        prompt = sample.get('llm_prompt', '')
        
        print(f"\n处理任务 {i+1}/{len(test_samples)}: {identifier}")
        print(f"提示长度: {len(prompt)} 字符")
        
        if not prompt:
            print("跳过: 空提示")
            continue
            
        try:
            # 调用LLM生成响应
            response = llm.generate_response(prompt)
            print(f"响应: {response[:200]}...")
            
            # 保存结果
            result = {
                "identifier": identifier,
                "model_name": "deepseek-chat",
                "llm_output": response
            }
            results.append(result)
            
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    # 保存结果
    output_dir = "./my_rsn_results/virtualhome/transition_modeling"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "my-rsn-model_outputs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"成功处理了 {len(results)} 个任务")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试transition modeling")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    
    args = parser.parse_args()
    test_transition_modeling(args.num_samples) 
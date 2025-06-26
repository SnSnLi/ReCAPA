#!/usr/bin/env python3
"""
简化的Transition Modeling测试脚本
"""

import json
import os
import sys
from pathlib import Path
import openai

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def simple_llm_call(prompt, api_key="sk-xxx", model="deepseek-chat", api_base="https://api.deepseek.com/v1"):
    """简单的LLM调用"""
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    
    system_prompt = """
You are an expert in state transition modeling for robotics. Given the current state (represented as predicates) and an action being executed, predict the resulting state after the action is performed.

IMPORTANT OUTPUT FORMAT:
- Your response must be a list of predicates representing the post-condition state
- Use proper predicate format: predicate_name(object.id)
- DO NOT use markdown code blocks or any other formatting
- Return only the list of predicates as a JSON array

Example correct format:
["soaked(rag.0)", "toggled_off(sink.82)", "clean(plate.1)"]

Your task is to analyze the given current state and action, then output the predicted state predicates after the action execution.
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content

def test_transition_modeling(num_samples=3):
    """测试transition modeling"""
    
    # 加载提示数据
    prompt_file = "./temp_prompts/virtualhome/generate_prompts/transition_modeling/helm_prompt.json"
    if not os.path.exists(prompt_file):
        print(f"错误: 找不到提示文件 {prompt_file}")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    print(f"加载了 {len(prompts)} 个transition modeling任务")
    
    # 测试前几个样本
    results = []
    test_samples = prompts[:num_samples]
    
    for i, sample in enumerate(test_samples):
        identifier = sample.get('identifier', f'sample_{i}')
        prompt = sample.get('llm_prompt', '')
        
        print(f"\n处理任务 {i+1}/{len(test_samples)}: {identifier}")
        
        if not prompt:
            print("跳过: 空提示")
            continue
            
        try:
            # 调用LLM生成响应
            response = simple_llm_call(prompt)
            print(f"响应: {response}")
            
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
    test_transition_modeling(3) 
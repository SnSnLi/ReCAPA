#!/usr/bin/env python3
"""
测试Transition Modeling F1指标的脚本
使用向量化提示模块增强上下文
"""

import json
import os
import sys
from pathlib import Path
import openai

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.llm_trajectory import LLMTrajectory


def test_transition_modeling_f1(num_samples=5, use_vectorized=True):
    """
    测试transition modeling并计算F1指标
    
    Args:
        num_samples: 测试样本数量
        use_vectorized: 是否使用向量化提示增强
    """
    
    # 加载提示数据
    prompt_file = "./temp_prompts/virtualhome/generate_prompts/transition_modeling/helm_prompt.json"
    if not os.path.exists(prompt_file):
        print(f"错误: 找不到提示文件 {prompt_file}")
        print("请先运行: eai-eval --mode generate_prompts --eval-type transition_modeling --dataset virtualhome --output-dir ./temp_prompts")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    print(f"加载了 {len(prompts)} 个transition modeling任务")
    print(f"将测试前 {num_samples} 个样本")
    print(f"向量化提示增强: {'启用' if use_vectorized else '禁用'}")
    print("=" * 50)
    
    # 初始化LLM
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("错误: 请设置环境变量 DEEPSEEK_API_KEY")
            return
            
        llm = LLMTrajectory(
            state_dim=64,  # VirtualHome典型状态维度
            action_dim=8,  # VirtualHome典型动作维度 
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
            enable_vectorized_prompts=use_vectorized
        )
        
        # 配置VirtualHome相关的元数据
        if use_vectorized:
            llm.update_vector_metadata(
                feature_names=[f"env_state_{i}" for i in range(64)],
                action_names=[f"action_{i}" for i in range(8)],
                state_range={"min": 0.0, "max": 1.0},
                action_range={"min": 0.0, "max": 1.0}
            )
    except Exception as e:
        print(f"LLM初始化失败: {e}")
        print("注意: 需要有效的API密钥")
        return
    
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
            if use_vectorized:
                # 使用向量化增强的提示
                response = llm.generate_trajectory_from_prompt(
                    prompt,
                    task_type="transition_modeling",
                    use_vectorized_enhancement=True
                )
            else:
                # 使用标准提示
                response = llm.generate_trajectory_from_prompt(
                    prompt,
                    task_type="transition_modeling"
                )
            
            print(f"响应长度: {len(response)} 字符")
            print(f"响应预览: {response[:100]}...")
            
            # 尝试解析JSON以验证格式
            try:
                parsed_response = json.loads(response)
                if isinstance(parsed_response, list):
                    print(f"✓ 有效的JSON列表，包含 {len(parsed_response)} 个谓词")
                else:
                    print(f"⚠ JSON格式正确但不是列表: {type(parsed_response)}")
            except json.JSONDecodeError:
                print("✗ JSON解析失败")
                # 尝试智能提取JSON
                if '{' in response and '}' in response:
                    start = response.find('[')
                    end = response.rfind(']') + 1
                    if start != -1 and end > start:
                        try:
                            extracted = response[start:end]
                            parsed_response = json.loads(extracted)
                            response = extracted
                            print(f"✓ 提取到有效JSON: {len(parsed_response)} 个谓词")
                        except:
                            print("✗ JSON提取也失败")
                            response = "[]"
                else:
                    response = "[]"
            
            # 保存结果
            result = {
                "identifier": identifier,
                "model_name": "deepseek-chat",
                "llm_output": response
            }
            results.append(result)
            
        except Exception as e:
            print(f"错误: {e}")
            # 添加失败的样本
            results.append({
                "identifier": identifier,
                "model_name": "deepseek-chat",
                "llm_output": "[]"  # 空响应
            })
            continue
    
    # 保存结果
    output_dir = "./my_rsn_results/virtualhome/transition_modeling"
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = "_vectorized" if use_vectorized else "_standard"
    output_file = os.path.join(output_dir, f"my-rsn-model_outputs{suffix}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"结果已保存到: {output_file}")
    print(f"成功处理了 {len(results)} 个任务")
    
    # 立即运行F1评估
    print(f"\n开始评估F1指标...")
    try:
        import subprocess
        eval_cmd = [
            "eai-eval",
            "--mode", "evaluate_results",
            "--eval-type", "transition_modeling", 
            "--dataset", "virtualhome",
            "--llm-response-path", output_dir,
            "--output-dir", f"./output/virtualhome/evaluate_results/transition_modeling/test{suffix}"
        ]
        
        print(f"执行命令: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✓ 评估成功完成")
            print("评估输出:")
            print(result.stdout)
        else:
            print(f"✗ 评估失败 (返回码: {result.returncode})")
            print("错误输出:")
            print(result.stderr)
            
    except Exception as e:
        print(f"运行评估时出错: {e}")
        print(f"请手动运行: eai-eval --mode evaluate_results --eval-type transition_modeling --dataset virtualhome --llm-response-path {output_dir}")
    
    return results


def compare_vectorized_vs_standard():
    """比较向量化提示和标准提示的性能"""
    print("=== 向量化提示 vs 标准提示性能比较 ===\n")
    
    # 测试标准提示
    print("1. 测试标准提示...")
    standard_results = test_transition_modeling_f1(num_samples=3, use_vectorized=False)
    
    print("\n" + "="*60 + "\n")
    
    # 测试向量化提示
    print("2. 测试向量化提示...")
    vectorized_results = test_transition_modeling_f1(num_samples=3, use_vectorized=True)
    
    print(f"\n{'='*60}")
    print("比较总结:")
    print(f"标准提示结果数量: {len(standard_results) if standard_results else 0}")
    print(f"向量化提示结果数量: {len(vectorized_results) if vectorized_results else 0}")
    print("\n请查看输出目录中的评估结果以比较F1分数。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试transition modeling F1指标")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--vectorized", action="store_true", help="使用向量化提示增强")
    parser.add_argument("--compare", action="store_true", help="比较向量化vs标准提示")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_vectorized_vs_standard()
    else:
        test_transition_modeling_f1(args.num_samples, args.vectorized) 
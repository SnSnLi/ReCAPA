#!/usr/bin/env python3
"""
将goal_interpretation任务的JSON对象格式转换为EAI要求的格式
"""

import json
import argparse
import re

def add_object_ids(predicate):
    """为谓词添加对象ID后缀"""
    # 匹配谓词模式：predicate(object1, object2) 或 predicate(object)
    pattern = r'(\w+)\(([^)]+)\)'
    match = re.match(pattern, predicate)
    if not match:
        return predicate
    
    pred_name, objects = match.groups()
    object_list = [obj.strip() for obj in objects.split(',')]
    
    # 为每个对象添加.1后缀（如果还没有的话）
    new_objects = []
    for obj in object_list:
        if '.' not in obj:
            new_objects.append(f"{obj}.1")
        else:
            new_objects.append(obj)
    
    return f"{pred_name}({', '.join(new_objects)})"

def convert_to_eai_format(input_file, output_file):
    """将JSON对象格式转换为EAI要求的格式"""
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换每个样本
    for item in data:
        if isinstance(item.get('llm_output'), dict) and 'structured_goal' in item['llm_output']:
            # 获取structured_goal列表
            structured_goal_list = item['llm_output']['structured_goal']
            
            # 为每个谓词添加对象ID
            processed_goals = []
            for goal in structured_goal_list:
                processed_goal = add_object_ids(goal)
                processed_goals.append(processed_goal)
            
            # 转换为EAI要求的格式：JSON对象包含structured_goal字段
            item['llm_output'] = {"structured_goal": processed_goals}
    
    # 保存转换后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"转换了 {len(data)} 个样本")
    
    # 显示转换示例
    if data:
        example = data[0]['llm_output']
        print(f"\n转换示例:")
        print(f"原始: {item.get('llm_output', 'N/A')}")
        print(f"转换后: {example}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将goal_interpretation任务的JSON对象格式转换为EAI要求的格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    
    args = parser.parse_args()
    convert_to_eai_format(args.input, args.output) 
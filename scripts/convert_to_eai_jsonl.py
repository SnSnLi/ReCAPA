#!/usr/bin/env python3
"""
将goal_interpretation任务的输出转换为EAI评测要求的.jsonl格式：
每行只包含{"structured_goal": [...]}，所有谓词带ID后缀。
"""
import json
import argparse
import re

def add_object_ids(predicate):
    pattern = r'(\w+)\(([^)]+)\)'
    match = re.match(pattern, predicate)
    if not match:
        return predicate
    pred_name, objects = match.groups()
    object_list = [obj.strip() for obj in objects.split(',')]
    new_objects = []
    for obj in object_list:
        if '.' not in obj:
            new_objects.append(f"{obj}.1")
        else:
            new_objects.append(obj)
    return f"{pred_name}({', '.join(new_objects)})"

def convert_to_eai_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in data:
            # 兼容llm_output为dict或字符串
            goals = []
            if isinstance(item.get('llm_output'), dict) and 'structured_goal' in item['llm_output']:
                goals = item['llm_output']['structured_goal']
            elif isinstance(item.get('llm_output'), str):
                # 兼容字符串格式，按行分割
                goals = [x.strip() for x in item['llm_output'].split('\n') if x.strip()]
            # 为每个谓词加ID
            processed_goals = [add_object_ids(g) for g in goals]
            # 只输出structured_goal字段
            out_obj = {"structured_goal": processed_goals}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
    print(f"转换完成！输出文件: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换为EAI评测要求的.jsonl格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径(.jsonl)")
    args = parser.parse_args()
    convert_to_eai_jsonl(args.input, args.output) 
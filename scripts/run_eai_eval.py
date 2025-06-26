import argparse
import json
import os
import sys
from tqdm import tqdm

# Add project root to Python path to allow importing from 'models'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_trajectory import LLMTrajectory
from agents.rsn_agent import RSNAgent

def main():
    """
    Main function to generate action sequences from a prompt file using an LLM.
    This script is intended to replace the previous debugging script and serves
    as the primary method for generating model outputs for EAI evaluation.
    """
    parser = argparse.ArgumentParser(description="Generate action sequences for EAI evaluation.")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-coder-6.7b-instruct", help="The name of the LLM to use.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the input JSON file containing prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated action sequences.")
    parser.add_argument("--task_type", type=str, default="action_sequencing", choices=["action_sequencing", "goal_interpretation", "subgoal_decomposition", "transition_modeling"], help="The type of task to generate predictions for.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing purposes).")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate trajectories to generate and select from.")
    args = parser.parse_args()

    # --- API Configuration ---
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = "https://api.deepseek.com"
    model_name = args.model

    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        sys.exit(1)

    print(f"Initializing generator with model: {model_name}")
    generator = LLMTrajectory(
        state_dim=1,  # Placeholder, not used for text generation but required by constructor
        action_dim=1, # Placeholder, not used for text generation but required by constructor
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
    )
    
    # Initialize RSN Agent for candidate selection (if using multiple candidates)
    rsn_agent = None
    if args.num_candidates > 1:
        print(f"Initializing RSN Agent for candidate selection (generating {args.num_candidates} candidates per prompt)")
        rsn_agent = RSNAgent(
            state_dim=512,  # Use reasonable values for text encoder
            action_dim=64,
            num_agents=1,
        )

    # --- Load Prompts ---
    try:
        with open(args.prompt_file, 'r') as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {args.pront_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.prompt_file}")
        sys.exit(1)
        
    # EAI prompts can be a list directly or a dict with a "prompts" key.
    if isinstance(prompts_data, dict):
        prompts = prompts_data.get("prompts", [])
    else:
        prompts = prompts_data

    # Limit samples if max_samples is specified
    if args.max_samples is not None and args.max_samples > 0:
        prompts = prompts[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples for testing.")

    # --- Generate Outputs ---
    all_outputs = []
    print(f"Generating outputs for {len(prompts)} prompts...")
    for idx, item in enumerate(tqdm(prompts, desc=f"Generating for {args.task_type}")):
        # The instruction can be under 'prompt', 'llm_prompt', or 'desc'
        instruction = item.get("prompt") or item.get("llm_prompt") or item.get("desc")
        if not instruction:
            print(f"Warning: Skipping item without a valid prompt field: {item}")
            continue

        print(f"\n=== 样本 {idx+1}/{len(prompts)} ===")
        print(f"ID: {item.get('instance_id') or item.get('identifier')}")
        print(f"输入提示 (前200字符): {instruction[:200]}...")
        print("正在生成候选答案...")

        try:
            # Generate multiple candidates if specified
            if args.num_candidates > 1 and rsn_agent is not None:
                candidates = []
                for i in range(args.num_candidates):
                    print(f"  生成候选 {i+1}/{args.num_candidates}...")
                    candidate_output = generator.generate_trajectory_from_prompt(
                        prompt_text=instruction,
                        task_type=args.task_type
                    )
                    candidates.append(candidate_output)
                    print(f"  候选 {i+1}: {candidate_output[:100]}...")
                
                print("  使用RSN选择最佳候选...")
                # Select the best candidate using RSN Agent
                raw_output = rsn_agent.select_best_trajectory(
                    task_description=instruction,
                    candidates=candidates
                )
                print(f"  选择的最佳候选: {raw_output[:100]}...")
            else:
                # Generate single output as before
                print("  生成单个输出...")
                raw_output = generator.generate_trajectory_from_prompt(
                    prompt_text=instruction,
                    task_type=args.task_type
                )
                print(f"  生成结果: {raw_output[:100]}...")
            
            # For action sequencing, the raw string is the expected output.
            # For other tasks, it might be a JSON string that needs parsing.
            output_content = raw_output
            if args.task_type != "action_sequencing":
                try:
                    # The model might wrap the JSON in markdown, so we clean it.
                    json_content = output_content
                    if "```" in json_content:
                        # Extract content between code blocks
                        parts = json_content.split('```')
                        if len(parts) >= 3:
                            json_content = parts[1].strip()
                            # Remove 'json' prefix if present
                            if json_content.startswith('json'):
                                json_content = json_content[4:].strip()
                    
                    parsed_json = json.loads(json_content)
                    print(f"  成功解析JSON: {str(parsed_json)[:100]}...")
                    
                    # 对于goal_interpretation，转换旧格式到新格式
                    if args.task_type == "goal_interpretation" and ("node_goals" in parsed_json or "node goals" in parsed_json):
                        structured_goal = []
                        
                        # 转换node_goals (处理两种key格式)
                        node_goals = parsed_json.get("node_goals", parsed_json.get("node goals", []))
                        for node_goal in node_goals:
                            if 'name' in node_goal and 'state' in node_goal:
                                structured_goal.append(f"{node_goal['state'].lower()}({node_goal['name']}.1)")
                        
                        # 转换edge_goals (处理两种key格式)
                        edge_goals = parsed_json.get("edge_goals", parsed_json.get("edge goals", []))
                        for edge_goal in edge_goals:
                            if 'from_name' in edge_goal and 'relation' in edge_goal and 'to_name' in edge_goal:
                                structured_goal.append(f"{edge_goal['relation'].lower()}({edge_goal['from_name']}.1, {edge_goal['to_name']}.2)")
                        
                        # 转换action_goals (处理两种key格式)
                        action_goals = parsed_json.get("action_goals", parsed_json.get("action goals", []))
                        for action_goal in action_goals:
                            if 'action' in action_goal:
                                structured_goal.append(f"{action_goal['action'].lower()}({action_goal.get('object', 'object')}.1)")
                        
                        # 转换为新格式
                        parsed_json = {"structured_goal": structured_goal}
                        print(f"  转换为新格式: {str(parsed_json)[:100]}...")
                    
                    output_content = parsed_json
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"  JSON解析失败，使用原始输出. 错误: {e}")
                    print(f"  原始内容: {raw_output[:200]}...")
                    # For transition modeling, if JSON parsing fails, try to extract just the output value
                    if args.task_type == "transition_modeling" and '"output":' in raw_output:
                        try:
                            # Try to extract the output value directly
                            start = raw_output.find('"output":') + 9
                            end = raw_output.rfind('}')
                            if start > 8 and end > start:
                                output_value = raw_output[start:end].strip()
                                if output_value.startswith('"') and output_value.endswith('"'):
                                    output_content = {"output": output_value[1:-1]}
                                    print(f"  成功提取output字段")
                        except:
                            pass

            # Structure the final output to match EAI evaluation format
            output_item = {
                "identifier": item.get("instance_id") or item.get("identifier"),
                "model_name": model_name,
                "llm_output": output_content
            }
            
            # 对于goal_interpretation，将structured_goal转换为字符串格式
            if args.task_type == "goal_interpretation" and isinstance(output_content, dict) and "structured_goal" in output_content:
                # 将谓词列表转换为字符串，用换行符分隔
                structured_goal_str = "\n".join(output_content["structured_goal"])
                output_item["llm_output"] = structured_goal_str
            
            all_outputs.append(output_item)
            
            print(f"最终输出: {str(output_content)[:150]}...")
            print("=" * 50)

        except Exception as e:
            print(f"FATAL: An unexpected error occurred for prompt '{instruction[:50]}...': {e}")
            all_outputs.append({
                "identifier": item.get("instance_id") or item.get("identifier"),
                "model_name": model_name,
                "llm_output": f"ERROR: {str(e)}"
            })
            
    # --- Save Results ---
    try:
        with open(args.output_file, 'w') as f:
            json.dump(all_outputs, f, indent=4)
        print(f"\nSuccessfully generated {len(all_outputs)} outputs.")
        print(f"Saved to {args.output_file}")
    except IOError as e:
        print(f"Error: Could not write to output file {args.output_file}: {e}")

if __name__ == "__main__":
    main() 
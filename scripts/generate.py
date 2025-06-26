import argparse
import json
import os
from tqdm import tqdm
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_trajectory import LLMTrajectory

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using an LLM based on input prompts.")
    parser.add_argument("--model", type=str, required=True, help="The name of the LLM to use (e.g., 'deepseek/deepseek-coder-6.7b-instruct').")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file with responses.")
    args = parser.parse_args()

    # Determine API key and base URL from model name and environment variables
    api_key = None
    api_base = None
    model_name = args.model.lower()

    if "gpt" in model_name:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = "https://api.openai.com/v1"
        print("Using OpenAI configuration.")
    elif "deepseek" in model_name:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        api_base = "https://api.deepseek.com"
        print("Using DeepSeek configuration.")
    
    if not api_key:
        print("Error: API key not found for the specified model.")
        print("Please set the OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    # Initialize the trajectory generator with all required parameters
    generator = LLMTrajectory(
        state_dim=512,  # Placeholder value for this script
        action_dim=64, # Placeholder value for this script
        api_key=api_key,
        model_name=args.model,
        api_base=api_base
    )

    # Load prompts from the input file
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # The EAI prompt files can have different structures.
    # Sometimes it's a dict with a "prompts" key, sometimes it's a list directly.
    if isinstance(data, dict):
        prompts = data.get("prompts", [])
    elif isinstance(data, list):
        prompts = data
    else:
        print(f"Error: Unexpected JSON structure in {args.input_file}. Expected a list or a dictionary with a 'prompts' key.")
        sys.exit(1)

    if not prompts:
         print(f"Error: No prompts found in {args.input_file}, or the list is empty.")
         sys.exit(1)


    all_responses = []
    
    print(f"Generating Trajectories for {len(prompts)} prompts...")
    for item in tqdm(prompts, desc="Generating Trajectories"):
        # The instruction can be under the key 'prompt' or 'llm_prompt'
        instruction_text = item.get("prompt") or item.get("llm_prompt")
        if not instruction_text:
            print(f"Warning: Skipping item with no 'prompt' or 'llm_prompt' field: {item}")
            continue

        try:
            # Generate a single trajectory/response for the instruction
            # The LLMTrajectoryGenerator will use the TRAJECTORY_PROMPT from prompts.py
            # We now call a different method for benchmark-style prompt generation
            raw_response = generator.generate_trajectory_from_prompt(
                prompt_text=instruction_text,
            )

            # The raw response is a string, which might be a JSON-formatted string.
            # We need to parse it to extract the list of actions/subgoals.
            try:
                # First try to extract JSON from markdown if present
                if raw_response.strip().startswith('```'):
                    # Extract content between markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw_response, re.DOTALL)
                    if json_match:
                        raw_response = json_match.group(1).strip()
                
                # Try to parse as JSON
                response_content = json.loads(raw_response)
                
                # If it's a dict (action sequencing format), keep it as is
                # If it's a list (subgoal format), keep it as is
                if not isinstance(response_content, (dict, list)):
                    response_content = [str(response_content)]
                    
            except json.JSONDecodeError as e:
                # If the model fails to produce valid JSON, try to extract meaningful content
                print(f"Warning: Failed to parse JSON for instruction '{instruction_text[:50]}...': {e}")
                print(f"Raw response: {raw_response[:200]}...")
                
                # Try to extract any list-like structure
                import re
                list_match = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
                if list_match:
                    try:
                        # Try to evaluate as Python list
                        response_content = eval(f"[{list_match.group(1)}]")
                    except:
                response_content = [raw_response]
                

            # EAI expects the output to be a dictionary for each prompt
            # The key for the task identifier is 'identifier'
            output_item = {
                "identifier": item.get("instance_id") or item.get("identifier"), # Accommodate both formats
                "model_name": args.model,
                "llm_output": response_content
            }
            all_responses.append(output_item)

        except Exception as e:
            print(f"Error generating trajectory for instruction '{instruction_text}': {e}")
            # Add a placeholder for the failed response
            all_responses.append({
                "identifier": item.get("instance_id") or item.get("identifier"),
                "model_name": args.model,
                "llm_output": f"ERROR: {str(e)}"
            })

    # Save all responses to the output file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(all_responses, f, indent=4)
        print(f"\nSuccessfully generated {len(all_responses)} responses.")
        print(f"Output saved to {args.output_file}")
    except IOError as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()

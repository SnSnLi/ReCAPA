import json
import re
from collections import Counter

def extract_unknown_primitives(file_path):
    """
    Parses an error_info.json file to extract all unique 'UnknownPrimitive' errors.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        A list of unique unknown primitives found in the file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return []

    unknown_primitives = []
    # Regex to find the unknown primitive from the error string, e.g., "Unknown primitive: GOTO"
    primitive_regex = re.compile(r"Unknown primitive: ([\w_]+)")

    for task_id, task_data in data.items():
        if not isinstance(task_data, dict):
            continue

        info = task_data.get('info')
        if not isinstance(info, str):
            continue
            
        # A simple string check is faster than parsing the tuple-like string
        if "'UnknownPrimitive'" in info:
            match = primitive_regex.search(info)
            if match:
                unknown_primitives.append(match.group(1))

    return unknown_primitives

def main():
    """
    Main function to analyze the error log and print the findings.
    """
    error_file = 'output/virtualhome/evaluate_results/subgoal_decomposition/my-deepseek-model-correct/virtualhome/evaluate_results/subgoal_decomposition/my-deepseek-model/error_info.json'
    
    print(f"--- Analyzing error log: {error_file} ---")
    
    primitives = extract_unknown_primitives(error_file)
    
    if not primitives:
        print("No 'UnknownPrimitive' errors found in the log file.")
        return

    print(f"\nFound a total of {len(primitives)} 'UnknownPrimitive' occurrences.")
    
    # Count the frequency of each primitive
    primitive_counts = Counter(primitives)
    
    print("\n--- Summary of Unknown Primitives (by frequency) ---")
    for primitive, count in primitive_counts.most_common():
        print(f"- {primitive}: {count} times")
        
    unique_primitives = sorted(list(primitive_counts.keys()))
    
    print("\n--- Complete List of Unique Unknown Primitives ---")
    print(unique_primitives)
    
    print("\nThis list represents the vocabulary the model *thinks* it can use, but is not recognized by the evaluator.")


if __name__ == "__main__":
    # Temporarily rename the original main to avoid conflict
    # and run our analysis instead.
    main() 
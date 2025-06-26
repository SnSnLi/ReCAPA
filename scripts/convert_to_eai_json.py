import json
import argparse
import os

def convert_to_eai_json_format(input_path: str, output_path: str):
    """
    Reads a model output file, converts the 'llm_output' field
    from a JSON object to the JSON string format expected by the EAI
    evaluator, and saves it to a new file.
    """
    print(f"Reading from: {input_path}")
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    converted_data = []
    for item in data:
        # Ensure the 'llm_output' field exists and is a dictionary (our current wrong format)
        if 'llm_output' in item and isinstance(item['llm_output'], dict):
            # Convert the dictionary to a JSON string
            item['llm_output'] = json.dumps(item['llm_output'])
        
        converted_data.append(item)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Writing converted data to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model output format for EAI evaluation.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to the input model outputs JSON file (in the wrong format).'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Path to save the converted output JSON file (in the correct format).'
    )
    args = parser.parse_args()
    
    convert_to_eai_json_format(args.input, args.output) 
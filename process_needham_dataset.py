"""
Process the Needham dataset to extract and format conversation samples.

This script processes raw conversation data from the Needham dataset and transforms
it into a standardized format suitable for evaluation. It:
- Formats messages for OpenAI API compatibility
- Computes eval_human scores from multiple author ratings
- Filters and cleans metadata fields
"""

import json
from typing import Any, Dict, List


def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single message to format it for OpenAI API.
    Removes the 'id' field and keeps only relevant fields.
    """
    processed = {
        "role": message["role"],
        "content": message["content"]
    }
    
    # Keep tool_calls if present (for assistant messages with tool calls)
    if "tool_calls" in message:
        processed["tool_calls"] = message["tool_calls"]
    
    # Keep tool_call_id if present (for tool response messages)
    if "tool_call_id" in message:
        processed["tool_call_id"] = message["tool_call_id"]
    
    return processed


def process_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process metadata to keep only specified fields and compute eval_human.
    """
    eval_author1 = metadata.get("eval_author1", 0)
    eval_author2 = metadata.get("eval_author2", 0)
    eval_human = round((eval_author1 + eval_author2) / 2, 4)
    
    processed_metadata = {
        "category": metadata.get("category"),
        "eval_category": metadata.get("eval_category"),
        "safety_category": metadata.get("safety_category"),
        "tools": metadata.get("tools"),
        "eval_type": metadata.get("eval_type"),
        "eval_human": eval_human
    }
    
    return processed_metadata


def should_include_sample(sample_id: str) -> bool:
    """
    Determine if a sample should be included in the processed dataset.
    
    Excludes:
    - All samples starting with "aider_commit"
    - All samples starting with "irl_cursor"
    - All samples starting with "continue_dev" except for specific IDs
    """
    # Exclude aider_commit samples
    if sample_id.startswith("aider_commit"):
        return False
    
    # Exclude irl_cursor samples
    if sample_id.startswith("irl_cursor"):
        return False
    
    # For continue_dev, only include specific samples
    if sample_id.startswith("continue_dev"):
        allowed_continue_dev = ["continue_dev:2", "continue_dev:8", "continue_dev:13", "continue_dev:22"]
        return sample_id in allowed_continue_dev
    
    # Include all other samples
    return True


def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single sample from the dataset.
    """
    processed_sample = {
        "id": sample["id"],
        "input": [process_message(msg) for msg in sample["input"]],
        "metadata": process_metadata(sample["metadata"])
    }
    
    return processed_sample


def main():
    input_file = "datasets/original_needham_dataset.json"
    output_file = "datasets/processed_cleaned_needham_dataset.json"
    
    print("Reading dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Filtering and processing {len(dataset)} samples...")
    # Filter samples based on ID criteria
    filtered_dataset = [sample for sample in dataset if should_include_sample(sample["id"])]
    print(f"Samples after filtering: {len(filtered_dataset)} (excluded {len(dataset) - len(filtered_dataset)} samples)")
    
    # Process the filtered samples
    processed_dataset = [process_sample(sample) for sample in filtered_dataset]
    
    print("Writing processed dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed dataset saved to: {output_file}")
    print(f"Total samples processed: {len(processed_dataset)}")


if __name__ == "__main__":
    main()


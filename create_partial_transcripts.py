"""
Create partial transcripts from a sample in processed_needham_dataset.json.

This script takes a sample ID and creates progressively longer partial transcripts.
Transcripts only end at valid stopping points:
- After an assistant message with no tool calls
- After all tool responses following an assistant message with tool calls

The output is formatted to be compatible with sample_integration.py.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy


def load_sample_by_id(sample_id: str, dataset_path: str = None) -> Dict[str, Any]:
    """
    Load a specific sample by ID from the processed dataset.
    
    Args:
        sample_id: The ID of the sample to load
        dataset_path: Path to the dataset file. If None, uses default path.
        
    Returns:
        Sample dictionary or None if not found
    """
    if dataset_path is None:
        script_dir = Path(__file__).parent
        dataset_path = script_dir / "datasets" / "processed_needham_dataset.json"
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Search for the sample with matching ID
    for sample in samples:
        if sample.get('id') == sample_id:
            return sample
    
    return None


def count_turns_in_sample(sample: Dict[str, Any]) -> int:
    """
    Count the total number of turns in a sample.
    A turn is defined as:
    - System message (if present)
    - User message
    - Assistant message (with optional tool_calls)
    - Tool response messages (one per tool_call)
    
    Args:
        sample: The sample dictionary
        
    Returns:
        Total number of turns
    """
    input_messages = sample.get('input', [])
    
    # We'll count message entries, treating each tool response as a turn
    turn_count = 0
    
    for message in input_messages:
        role = message.get('role')
        
        if role in ['system', 'user']:
            turn_count += 1
        elif role == 'assistant':
            turn_count += 1
            # Each tool call is a separate turn (the tool response follows)
        elif role == 'tool':
            turn_count += 1
    
    return turn_count


def create_partial_transcript(sample: Dict[str, Any], num_turns: int) -> Dict[str, Any]:
    """
    Create a partial transcript with a specified number of turns.
    
    Args:
        sample: The original sample dictionary
        num_turns: Number of turns to include (must be at least 2 for system + first user message)
        
    Returns:
        New sample dictionary with partial transcript
    """
    input_messages = sample.get('input', [])
    
    # Create a deep copy of the sample to modify
    partial_sample = deepcopy(sample)
    
    # Track turns included
    turns_included = 0
    partial_input = []
    
    for i, message in enumerate(input_messages):
        if turns_included >= num_turns:
            break
            
        role = message.get('role')
        
        if role in ['system', 'user']:
            partial_input.append(message)
            turns_included += 1
        elif role == 'assistant':
            partial_input.append(message)
            turns_included += 1
        elif role == 'tool':
            partial_input.append(message)
            turns_included += 1
    
    # Update the partial sample
    partial_sample['input'] = partial_input
    
    # Update the ID to reflect the length
    original_id = sample.get('id', 'unknown')
    partial_sample['id'] = f"{original_id}_turns_{num_turns}"
    
    return partial_sample


def find_valid_stopping_points(sample: Dict[str, Any]) -> List[int]:
    """
    Find valid stopping points where partial transcripts can end.
    Valid stopping points are after:
    - An assistant message with no tool_calls
    - All tool responses following an assistant message with tool_calls
    
    Args:
        sample: The original sample dictionary
        
    Returns:
        List of message indices (0-based) where transcripts can validly end
    """
    input_messages = sample.get('input', [])
    valid_stopping_points = []
    
    i = 0
    while i < len(input_messages):
        message = input_messages[i]
        role = message.get('role')
        
        if role == 'assistant':
            tool_calls = message.get('tool_calls', [])
            
            if not tool_calls:
                # Assistant message with no tool calls - valid stopping point
                valid_stopping_points.append(i)
                i += 1
            else:
                # Assistant message with tool calls - need to include all tool responses
                num_tool_calls = len(tool_calls)
                # Skip to after all the tool responses
                j = i + 1
                tool_responses_found = 0
                
                while j < len(input_messages) and tool_responses_found < num_tool_calls:
                    if input_messages[j].get('role') == 'tool':
                        tool_responses_found += 1
                    j += 1
                
                # If we found all tool responses, this is a valid stopping point
                if tool_responses_found == num_tool_calls:
                    valid_stopping_points.append(j - 1)  # Last tool response
                    i = j
                else:
                    i += 1
        else:
            i += 1
    
    return valid_stopping_points


def create_all_partial_transcripts(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create all partial transcripts from minimum (system + user) to full transcript.
    Only creates transcripts that end with an assistant response or complete tool use.
    
    Args:
        sample: The original sample dictionary
        
    Returns:
        List of partial transcript samples, ordered from shortest to longest
    """
    input_messages = sample.get('input', [])
    valid_stopping_points = find_valid_stopping_points(sample)
    
    partial_transcripts = []
    
    # Create partial transcripts for each valid stopping point
    for stop_index in valid_stopping_points:
        # Create a deep copy of the sample to modify
        partial_sample = deepcopy(sample)
        
        # Include all messages up to and including the stopping point
        partial_input = input_messages[:stop_index + 1]
        partial_sample['input'] = partial_input
        
        # Update the ID to reflect the length
        original_id = sample.get('id', 'unknown')
        num_messages = len(partial_input)
        partial_sample['id'] = f"{original_id}_turns_{num_messages}"
        
        partial_transcripts.append(partial_sample)
    
    return partial_transcripts


def save_partial_transcripts(
    partial_transcripts: List[Dict[str, Any]], 
    original_sample_id: str
) -> str:
    """
    Save all partial transcripts to a JSON file in the partial_transcripts folder.
    
    Args:
        partial_transcripts: List of partial transcript samples
        original_sample_id: Original sample ID (used for generating filename)
        
    Returns:
        Path to the saved file
    """
    # Create partial_transcripts directory if it doesn't exist
    script_dir = Path(__file__).parent
    output_dir = script_dir / "partial_transcripts"
    output_dir.mkdir(exist_ok=True)
    
    # Clean the sample ID for use in filename
    safe_id = original_sample_id.replace(':', '_').replace('/', '_')
    output_path = output_dir / f"partial_transcripts_{safe_id}.json"
    
    # Save as JSON (format compatible with sample_integration.py)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(partial_transcripts, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(partial_transcripts)} partial transcripts to: {output_path}")
    return str(output_path)


def create_partial_transcripts_for_sample(sample_id: str, dataset_path: str = None) -> str:
    """
    Create partial transcripts for a given sample ID.
    
    Args:
        sample_id: The ID of the sample to process
        dataset_path: Path to the dataset file (optional)
        
    Returns:
        Path to the saved output file
    """
    # Load the sample
    print(f"Loading sample with ID: {sample_id}")
    sample = load_sample_by_id(sample_id, dataset_path)
    
    if sample is None:
        raise ValueError(f"Sample with ID '{sample_id}' not found in dataset")
    
    print(f"✓ Found sample: {sample_id}")
    
    # Count total turns
    total_turns = count_turns_in_sample(sample)
    print(f"✓ Sample has {total_turns} total turns")
    
    # Create all partial transcripts
    print("Creating partial transcripts...")
    partial_transcripts = create_all_partial_transcripts(sample)
    print(f"✓ Created {len(partial_transcripts)} partial transcripts")
    
    # Save to file
    output_file = save_partial_transcripts(partial_transcripts, sample_id)
    
    print(f"\n✓ Done! Partial transcripts saved to: {output_file}")
    print(f"  - Each transcript ends with an assistant response or complete tool use")
    print(f"  - Each transcript preserves original metadata")
    print(f"  - IDs are modified to include message count (e.g., '{sample_id}_turns_5')")
    print(f"  - Output format is compatible with sample_integration.py")
    
    return output_file


if __name__ == "__main__":
    # Example usage
    sample_id = "swe_gym:5893"  # Change this to the sample ID you want to process
    create_partial_transcripts_for_sample(sample_id)


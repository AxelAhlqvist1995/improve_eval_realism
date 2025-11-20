#!/usr/bin/env python3
"""
Extract prompts from Inspect AI eval logs in OpenAI API format.
The messages can be used directly with OpenAI API without modification.
"""

import json
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def extract_content_parts(content: Any) -> tuple[str, str]:
    """
    Extract text and reasoning from content.
    Returns: (text_content, reasoning_content)
    """
    text_parts = []
    reasoning_parts = []
    
    if isinstance(content, str):
        return content, ""
    elif isinstance(content, list):
        for item in content:
            if hasattr(item, 'text') and item.text:
                text_parts.append(item.text)
            if hasattr(item, 'reasoning') and item.reasoning:
                reasoning_parts.append(item.reasoning)
            if isinstance(item, dict):
                if 'text' in item and item['text']:
                    text_parts.append(item['text'])
                if 'reasoning' in item and item['reasoning']:
                    reasoning_parts.append(item['reasoning'])
        
        text = '\n'.join(text_parts) if text_parts else ""
        reasoning = '\n'.join(reasoning_parts) if reasoning_parts else ""
        return text, reasoning
    else:
        return str(content), ""


def extract_prompts_minimal(log_file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract prompts in OpenAI API format.
    
    Args:
        log_file_path: Path to the .eval log file
        
    Returns:
        List of samples with id and messages in OpenAI API format
    """
    log = read_eval_log(log_file_path)
    extracted_data = []
    
    # Get dataset name from the log
    dataset_name = log.eval.dataset.name if hasattr(log, 'eval') and hasattr(log.eval, 'dataset') and hasattr(log.eval.dataset, 'name') else "oversight_subversion"
    
    # Count epochs per sample ID to determine if we need to add epoch suffix
    sample_id_to_epochs = {}
    for sample in log.samples:
        sample_id = sample.id if hasattr(sample, 'id') else None
        epoch = sample.epoch if hasattr(sample, 'epoch') else 1
        
        if sample_id not in sample_id_to_epochs:
            sample_id_to_epochs[sample_id] = []
        sample_id_to_epochs[sample_id].append(epoch)
    
    for idx, sample in enumerate(log.samples):
        messages = []
        
        if hasattr(sample, 'messages') and sample.messages:
            for msg in sample.messages:
                # Handle system, user, and assistant messages
                if msg.role in ['system', 'user', 'assistant']:
                    text_content, reasoning_content = extract_content_parts(msg.content)
                    
                    # Build content string (only use text, exclude reasoning)
                    # Reasoning is not shown to users in real scenarios
                    content = text_content if text_content else None
                    
                    # Build message in OpenAI format
                    message = {"role": msg.role}
                    
                    # Add content if present
                    if content:
                        message["content"] = content
                    
                    # Add tool calls if present (for assistant messages) in OpenAI format
                    if msg.role == 'assistant' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls = []
                        for tc in msg.tool_calls:
                            # Convert arguments to JSON string if it's a dict
                            args = tc.arguments
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            elif not isinstance(args, str):
                                args = str(args)
                            
                            tool_call = {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function,
                                    "arguments": args
                                }
                            }
                            tool_calls.append(tool_call)
                        message["tool_calls"] = tool_calls
                    
                    # OpenAI requires content to be present or null for assistant messages
                    if msg.role == 'assistant' and 'content' not in message:
                        message["content"] = None
                    
                    # Only add if there's actual content or tool calls
                    if "content" in message or "tool_calls" in message:
                        messages.append(message)
                
                # Handle tool response messages in OpenAI format
                elif msg.role == 'tool':
                    tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    
                    # Get tool_call_id from the message
                    tool_call_id = getattr(msg, 'tool_call_id', None)
                    
                    if tool_call_id:
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_content
                        }
                        messages.append(tool_response)
        
        # Get sample ID and epoch from the sample
        sample_id = sample.id if hasattr(sample, 'id') else idx
        epoch = sample.epoch if hasattr(sample, 'epoch') else 1
        
        # Build the ID: dataset_name:sample_id or dataset_name:sample_id_epoch
        # Only add epoch suffix if there are multiple epochs for this sample ID
        has_multiple_epochs = len(sample_id_to_epochs.get(sample_id, [1])) > 1
        if has_multiple_epochs:
            id_string = f"{dataset_name}:{sample_id}_{epoch}"
        else:
            id_string = f"{dataset_name}:{sample_id}"
        
        sample_data = {
            "id": id_string,
            "prompts": messages,
            "metadata": {
                "category": dataset_name,
                "eval_category": True,
                "safety_category": True,
                "eval_type": "alignment",
                "human_eval_probability": None
            }
        }
        
        extracted_data.append(sample_data)
    
    return extracted_data


def load_samples_from_file(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load samples from either a .eval file or .json file.
    
    Handles both:
    - .eval files: Processed through extract_prompts_minimal()
    - Inspect AI JSON files: With "eval" and "samples" keys, extracts dataset name and handles epochs
    - Regular JSON files: Arrays of samples in standard format
    
    Args:
        file_path: Path to the input file (.eval or .json)
        
    Returns:
        List of sample dictionaries with properly formatted IDs and data
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.json':
        print(f"Loading samples from JSON file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is an Inspect AI structured JSON (with "eval" and "samples" keys)
        if isinstance(data, dict) and 'eval' in data and 'samples' in data:
            # Extract dataset name from eval.dataset.name
            dataset_name = data.get('eval', {}).get('dataset', {}).get('name', 'oversight_subversion')
            raw_samples = data['samples']
            
            # Count epochs per sample ID to determine if we need to add epoch suffix
            sample_id_to_epochs = {}
            for sample in raw_samples:
                sample_id = sample.get('id')
                epoch = sample.get('epoch', 1)
                
                if sample_id not in sample_id_to_epochs:
                    sample_id_to_epochs[sample_id] = []
                sample_id_to_epochs[sample_id].append(epoch)
            
            # Process samples to rename them with dataset name and proper epoch handling
            samples = []
            for sample in raw_samples:
                sample_id = sample.get('id')
                epoch = sample.get('epoch', 1)
                
                # Build the ID: dataset_name:sample_id or dataset_name:sample_id_epoch
                # Only add epoch suffix if there are multiple epochs for this sample ID
                has_multiple_epochs = len(sample_id_to_epochs.get(sample_id, [1])) > 1
                if has_multiple_epochs:
                    id_string = f"{dataset_name}:{sample_id}_{epoch}"
                else:
                    id_string = f"{dataset_name}:{sample_id}"
                
                # Create new sample with renamed ID
                # Note: In Inspect AI JSON, 'messages' contains the full conversation,
                # while 'input' is just the initial user input string.
                # We need 'input' to be the messages array for extract_conversation_to_string.
                processed_sample = {
                    'id': id_string,
                    'input': sample.get('messages', sample.get('input', sample.get('prompts', []))),
                    'metadata': sample.get('metadata', {
                        'category': dataset_name,
                        'eval_category': True,
                        'safety_category': True,
                        'eval_type': 'alignment',
                        'human_eval_probability': None
                    })
                }
                
                samples.append(processed_sample)
            
            print(f"✓ Loaded {len(samples)} samples from Inspect AI JSON")
            print(f"✓ Dataset name: {dataset_name}")
        else:
            # Regular JSON with array of samples (already in the right format)
            samples = data if isinstance(data, list) else [data]
            print(f"✓ Loaded {len(samples)} samples from JSON")
    else:
        print(f"Extracting samples from .eval file: {file_path}")
        samples = extract_prompts_minimal(str(file_path))
        print(f"✓ Extracted {len(samples)} samples from .eval log")
    
    # Normalize the message key: convert 'prompts' to 'input' if needed
    # (extract_prompts_minimal uses 'prompts', but the rest of the codebase expects 'input')
    for sample in samples:
        if 'prompts' in sample and 'input' not in sample:
            sample['input'] = sample.pop('prompts')
    
    return samples


def main():
    # Get the script directory and navigate to logs
    script_dir = Path(__file__).parent.parent  # Go up to project root
    log_file = script_dir / "logs" / "2025-10-08T17-45-27+03-00_oversight-subversion_D3qfJ6aY93NN2jhbPYgQ6W.eval"
    
    print(f"Processing: {log_file}")
    
    # Extract prompts in minimal format
    extracted_prompts = extract_prompts_minimal(str(log_file))
    
    # Save to JSON
    output_file = log_file.with_suffix('').with_name(log_file.stem + "_prompts_minimal.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(extracted_prompts)} samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()


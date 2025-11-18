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
        
        sample_data = {
            "id": "oversight_subversion:" + str(idx),
            "prompts": messages,
            "metadata": {
                "category": "oversight_subversion",
                "eval_category": True,
                "safety_category": True,
                "eval_type": "alignment",
                "human_eval_probability": None
            }
        }
        
        extracted_data.append(sample_data)
    
    return extracted_data


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


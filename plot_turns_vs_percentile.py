#!/usr/bin/env python3
"""
Create a scatter plot showing the relationship between percentile and number of turns
using Apollo leaderboard placement data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_turns_from_sample_id(sample_id):
    """Extract number of turns from sample ID like 'openhands:crosscoders_final_turns_1'"""
    match = re.search(r'turns_(\d+)', sample_id)
    if match:
        return int(match.group(1))
    return None

def load_apollo_data(file_path):
    """Load Apollo leaderboard data and extract turns and percentiles"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    turns = []
    percentiles = []
    
    for sample in data['samples']:
        sample_id = sample['new_sample_id']
        percentile = sample['leaderboard_percentile']
        
        num_turns = extract_turns_from_sample_id(sample_id)
        if num_turns is not None:
            turns.append(num_turns)
            percentiles.append(percentile)
    
    return turns, percentiles

def main():
    # Load data
    apollo_file = 'apollo_leaderboard_placement/swe_gym:5893_grok_4_fast.json'
    turns, percentiles = load_apollo_data(apollo_file)
    
    print(f"Loaded {len(turns)} samples from Apollo data")
    print(f"Turns range: {min(turns)} to {max(turns)}")
    print(f"Percentiles range: {min(percentiles):.1f} to {max(percentiles):.1f}")
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(percentiles, turns, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add labels and title
    plt.xlabel('Leaderboard Percentile', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Turns', fontsize=12, fontweight='bold')
    plt.title('Relationship between Leaderboard Percentile and Number of Turns', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Calculate and display correlation
    correlation = np.corrcoef(percentiles, turns)[0, 1]
    
    # Add statistics
    median_percentile = np.median(percentiles)
    plt.axvline(x=median_percentile, color='orange', linestyle=':', alpha=0.7, linewidth=1, 
                label=f'Median: {median_percentile:.1f}th')
    
    plt.legend(loc='lower left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = 'apollo_turns_vs_percentile_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Display statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(turns)}")
    print(f"  Turns range: [{min(turns)}, {max(turns)}]")
    print(f"  Percentile range: [{min(percentiles):.1f}, {max(percentiles):.1f}]")
    print(f"  Median percentile: {median_percentile:.1f}th")
    print(f"  Correlation (percentile vs turns): {correlation:.3f}")
    print(f"  Samples below 50th percentile: {sum(1 for p in percentiles if p < 50)} ({100*sum(1 for p in percentiles if p < 50)/len(percentiles):.1f}%)")
    print(f"  Samples above 50th percentile: {sum(1 for p in percentiles if p >= 50)} ({100*sum(1 for p in percentiles if p >= 50)/len(percentiles):.1f}%)")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

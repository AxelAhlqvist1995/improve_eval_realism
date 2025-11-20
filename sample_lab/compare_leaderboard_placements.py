#!/usr/bin/env python3
"""
Compare samples across multiple evaluation files using dumbbell plots.

This script takes multiple JSON files from samples_mapped_to_leaderboard and creates
a dumbbell plot showing how samples with identical IDs perform across different evaluations.
Each evaluation file is color-coded for easy comparison.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_evaluation_file(file_path: Path) -> Dict:
    """Load a JSON evaluation file and return its data."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_sample_data(eval_data: Dict) -> Dict[str, Dict]:
    """
    Extract sample IDs and their metrics from an evaluation file.
    
    Returns:
        Dict mapping sample_id to {theta, sigma, percentile}
    """
    samples = {}
    
    # Get all theta values to calculate percentiles
    all_thetas = [s['final_theta'] for s in eval_data['samples']]
    
    for sample in eval_data['samples']:
        sample_id = sample['new_sample_id']
        final_theta = sample['final_theta']
        
        # Calculate percentile rank (higher theta = higher percentile)
        percentile = (np.sum(np.array(all_thetas) < final_theta) / len(all_thetas)) * 100
        
        samples[sample_id] = {
            'theta': final_theta,
            'sigma': sample['final_sigma'],
            'percentile': percentile
        }
    
    return samples


def find_common_samples(eval_files_data: Dict[str, Dict]) -> List[str]:
    """
    Find sample IDs that appear in all evaluation files.
    
    Args:
        eval_files_data: Dict mapping file_label to sample_data dict
        
    Returns:
        List of common sample IDs
    """
    if not eval_files_data:
        return []
    
    # Start with samples from first file
    common_ids = set(next(iter(eval_files_data.values())).keys())
    
    # Intersect with all other files
    for sample_data in eval_files_data.values():
        common_ids &= set(sample_data.keys())
    
    return sorted(list(common_ids))


def create_dumbbell_plot(
    eval_files_data: Dict[str, Dict],
    common_samples: List[str],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Create a dumbbell plot comparing samples across evaluation files.
    
    Args:
        eval_files_data: Dict mapping file_label to sample_data dict
        common_samples: List of sample IDs to plot
        output_path: Path to save the plot
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    if not common_samples:
        print("No common samples found across all files.")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different files (using a colorblind-friendly palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    file_labels = list(eval_files_data.keys())
    n_files = len(file_labels)
    
    if n_files > len(colors):
        # Generate more colors if needed
        colors = plt.cm.tab20(np.linspace(0, 1, n_files))
    
    # Sort samples by their average value across files
    sample_averages = {}
    for sample_id in common_samples:
        values = [eval_files_data[label][sample_id]['percentile'] for label in file_labels]
        sample_averages[sample_id] = np.mean(values)
    
    sorted_samples = sorted(common_samples, key=lambda x: sample_averages[x], reverse=True)
    
    # Plot each sample
    y_positions = range(len(sorted_samples))
    
    for i, sample_id in enumerate(sorted_samples):
        # Get all values for this sample across files
        values = [eval_files_data[label][sample_id]['percentile'] for label in file_labels]
        
        # Draw line connecting min and max values
        min_val = min(values)
        max_val = max(values)
        ax.plot([min_val, max_val], [i, i], 'k-', alpha=0.3, linewidth=1, zorder=1)
        
        # Plot points for each file
        for j, (label, value) in enumerate(zip(file_labels, values)):
            ax.scatter(value, i, color=colors[j], s=100, alpha=0.7, 
                      edgecolors='white', linewidth=1.5, zorder=2, label=label if i == 0 else "")
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_samples, fontsize=8)
    
    ax.set_xlabel('Percentile Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample ID', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(f'Sample Performance Comparison Across Evaluations\n({len(sorted_samples)} common samples)', 
                    fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with file labels
    handles = [mpatches.Patch(color=colors[i], label=label) 
               for i, label in enumerate(file_labels)]
    ax.legend(handles=handles, loc='best', framealpha=0.9, fontsize=9)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()


def compare_evaluation_files(
    file_paths: List[Path],
    output_name: Optional[str] = None,
    title: Optional[str] = None
) -> Tuple[Path, int]:
    """
    Main function to compare evaluation files and create dumbbell plot.
    
    Args:
        file_paths: List of paths to JSON evaluation files (can be just filenames)
        output_name: Optional name for the output plot (without extension)
        title: Optional title for the plot
        
    Returns:
        Tuple of (output_path, number of common samples)
    """
    # Set default directories
    script_dir = Path(__file__).parent
    default_samples_dir = script_dir / "samples_mapped_to_leaderboard"
    output_dir = script_dir / "plots" / "eval_file_comparison"
    
    # Generate output filename
    if output_name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_name = f"comparison_{timestamp}"
    
    output_path = output_dir / f"{output_name}.png"
    
    # Resolve file paths - if just a filename/relative path, look in samples_mapped_to_leaderboard
    resolved_paths = []
    for file_path in file_paths:
        if file_path.exists():
            # Full path that exists
            resolved_paths.append(file_path)
        else:
            # Try as relative path within samples_mapped_to_leaderboard
            alt_path = default_samples_dir / str(file_path)
            if alt_path.exists():
                resolved_paths.append(alt_path)
            else:
                print(f"Warning: File not found: {file_path}")
                print(f"  Also tried: {alt_path}")
                continue
    
    # Load all evaluation files
    eval_files_data = {}
    for file_path in resolved_paths:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Loading: {file_path.name}")
        eval_data = load_evaluation_file(file_path)
        sample_data = extract_sample_data(eval_data)
        
        # Use file stem as label
        label = file_path.stem
        eval_files_data[label] = sample_data
    
    if not eval_files_data:
        raise ValueError("No valid evaluation files loaded.")
    
    # Find common samples
    common_samples = find_common_samples(eval_files_data)
    print(f"\nFound {len(common_samples)} common samples across all files.")
    
    if not common_samples:
        print("Warning: No common samples found. Cannot create plot.")
        return output_path, 0
    
    # Create the plot
    create_dumbbell_plot(
        eval_files_data=eval_files_data,
        common_samples=common_samples,
        output_path=output_path,
        title=title
    )
    
    return output_path, len(common_samples)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare samples across multiple evaluation files using dumbbell plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two evaluation files (automatically looks in samples_mapped_to_leaderboard/)
  python compare_eval_files.py file1.json file2.json
  
  # Compare with custom output name
  python compare_eval_files.py file1.json file2.json -o my_comparison
  
  # Add custom title
  python compare_eval_files.py file1.json file2.json -t "My Comparison"
  
  # You can also use full paths if files are elsewhere
  python compare_eval_files.py /path/to/file1.json /path/to/file2.json
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        type=str,
        help='JSON evaluation filenames (or full paths) to compare'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output filename (without extension). Defaults to timestamp.'
    )
    
    parser.add_argument(
        '-t', '--title',
        type=str,
        default=None,
        help='Custom title for the plot'
    )
    
    args = parser.parse_args()
    
    # Convert file paths
    file_paths = [Path(f) for f in args.files]
    
    # Run comparison
    try:
        output_path, n_common = compare_evaluation_files(
            file_paths=file_paths,
            output_name=args.output,
            title=args.title
        )
        
        print(f"\n✓ Comparison complete!")
        print(f"  Plot: {output_path}")
        print(f"  Common samples: {n_common}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


"""
Utility script to aggregate existing individual integration result files.

This script scans the samples_mapped_to_leaderboard directory for individual result files
and combines them into a single comprehensive aggregate file.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def find_integration_result_files(results_dir: Path) -> List[Path]:
    """
    Find all individual integration result files in the directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        List of paths to integration result files
    """
    # Look for any .json files except aggregate files
    files = [f for f in sorted(results_dir.glob("*.json")) 
             if not f.stem.startswith("aggregate_")]
    return files


def load_integration_results(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Load all integration results from individual files.
    
    Args:
        file_paths: List of paths to integration result files
        
    Returns:
        List of integration result dictionaries
    """
    results = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"⚠ Warning: Could not load {file_path.name}: {e}")
    
    return results


def aggregate_results(
    results: List[Dict[str, Any]],
    output_dir: Path
) -> str:
    """
    Aggregate individual results into a single comprehensive file.
    
    Args:
        results: List of integration result dictionaries
        output_dir: Directory to save aggregate file
        
    Returns:
        Path to saved aggregate file
    """
    if not results:
        raise ValueError("No results to aggregate")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"aggregate_integration_results_{timestamp}.json"
    output_path = output_dir / filename
    
    # Sort results by sample_id for consistency
    sorted_results = sorted(results, key=lambda r: r.get('new_sample_id', ''))
    
    # Create aggregate structure
    aggregate = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples": len(results),
            "total_comparisons": sum(r.get('num_comparisons', 0) for r in results),
            "average_comparisons": sum(r.get('num_comparisons', 0) for r in results) / len(results) if results else 0,
            "average_theta": sum(r.get('final_theta', 0) for r in results) / len(results) if results else 0,
            "average_sigma": sum(r.get('final_sigma', 0) for r in results) / len(results) if results else 0,
        },
        "samples": sorted_results
    }
    
    # Add convergence statistics
    converged = sum(1 for r in results if 'threshold' in r.get('convergence_reason', '').lower())
    aggregate["metadata"]["convergence_statistics"] = {
        "reached_threshold": converged,
        "hit_max_comparisons": len(results) - converged,
        "convergence_rate": f"{100 * converged / len(results):.1f}%" if results else "N/A"
    }
    
    # Add leaderboard statistics if available
    if results and 'leaderboard_rank' in results[0]:
        ranks = [r['leaderboard_rank'] for r in results if 'leaderboard_rank' in r]
        percentiles = [r['leaderboard_percentile'] for r in results if 'leaderboard_percentile' in r]
        
        if ranks and percentiles:
            aggregate["metadata"]["leaderboard_statistics"] = {
                "leaderboard_size": results[0].get('leaderboard_total', 0),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "median_rank": sorted(ranks)[len(ranks)//2],
                "best_percentile": max(percentiles),
                "worst_percentile": min(percentiles),
                "median_percentile": sorted(percentiles)[len(percentiles)//2]
            }
    
    # Save JSON with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, indent=2)
    
    return str(output_path)


def main():
    """
    Main function to aggregate existing integration result files.
    """
    # Get samples_mapped_to_leaderboard directory (in sample_lab)
    script_dir = Path(__file__).parent
    results_dir = script_dir / "samples_mapped_to_leaderboard"
    
    if not results_dir.exists():
        print(f"❌ Error: Directory not found at {results_dir}")
        return
    
    print("="*80)
    print("AGGREGATING EXISTING INTEGRATION RESULTS")
    print("="*80)
    print(f"Scanning directory: {results_dir}\n")
    
    # Find all integration result files
    file_paths = find_integration_result_files(results_dir)
    
    if not file_paths:
        print("ℹ No integration result files found.")
        print("  Looking for .json files (excluding aggregate_*.json)")
        print(f"  In directory: {results_dir}")
        return
    
    print(f"Found {len(file_paths)} result files:")
    for file_path in file_paths[:5]:  # Show first 5
        print(f"  - {file_path.name}")
    if len(file_paths) > 5:
        print(f"  ... and {len(file_paths) - 5} more")
    
    print("\nLoading results...")
    results = load_integration_results(file_paths)
    
    if not results:
        print("❌ Error: No results could be loaded from files")
        return
    
    print(f"✓ Successfully loaded {len(results)} results")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    total_comparisons = sum(r.get('num_comparisons', 0) for r in results)
    avg_comparisons = total_comparisons / len(results) if results else 0
    avg_theta = sum(r.get('final_theta', 0) for r in results) / len(results) if results else 0
    avg_sigma = sum(r.get('final_sigma', 0) for r in results) / len(results) if results else 0
    
    print(f"Total samples: {len(results)}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Average comparisons per sample: {avg_comparisons:.1f}")
    print(f"Average final theta: {avg_theta:.2f}")
    print(f"Average final sigma: {avg_sigma:.2f}")
    
    # Convergence statistics
    converged = sum(1 for r in results if 'threshold' in r.get('convergence_reason', '').lower())
    print(f"\nConvergence:")
    print(f"  Reached threshold: {converged}/{len(results)}")
    print(f"  Hit max comparisons: {len(results) - converged}/{len(results)}")
    print(f"  Convergence rate: {100 * converged / len(results):.1f}%")
    
    # Leaderboard statistics if available
    if results and 'leaderboard_rank' in results[0]:
        ranks = [r['leaderboard_rank'] for r in results if 'leaderboard_rank' in r]
        percentiles = [r['leaderboard_percentile'] for r in results if 'leaderboard_percentile' in r]
        
        if ranks and percentiles:
            print(f"\nLeaderboard placement:")
            print(f"  Best rank: #{min(ranks)} ({max(percentiles)}th percentile)")
            print(f"  Worst rank: #{max(ranks)} ({min(percentiles)}th percentile)")
            print(f"  Median rank: #{sorted(ranks)[len(ranks)//2]} ({sorted(percentiles)[len(percentiles)//2]}th percentile)")
    
    # Aggregate and save
    print("\n" + "="*80)
    print("SAVING AGGREGATE RESULTS")
    print("="*80)
    output_path = aggregate_results(results, results_dir)
    file_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"✓ Aggregate results saved to:")
    print(f"  {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()


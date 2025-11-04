"""
Adaptive integration algorithm for Bradley-Terry rating system.

This module implements an adaptive algorithm to efficiently integrate new samples
into an existing calibration set using strategic pairwise comparisons.
"""

import json
import math
import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Callable
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from sample_comparison import extract_conversation_to_string, compare_transcripts_realism
from extract_prompts_from_inspect_log import extract_prompts_minimal
from bradley_terry import bradley_terry_mm, INITIAL_SIGMA


def load_calibration_ratings(results_path: str = None) -> Dict[str, Tuple[float, float]]:
    """
    Load calibration samples with their current ratings from Bradley-Terry results.
    
    Args:
        results_path: Path to the Bradley-Terry results JSON file. 
                     If None, uses default path.
        
    Returns:
        Dictionary mapping sample_id to (theta, sigma) tuple
        
    Example:
        {"sample_id": (theta, sigma), ...}
    """
    if results_path is None:
        # Default path relative to this script
        script_dir = Path(__file__).parent
        results_path = script_dir / "results" / "bradley_terry_results_2025-10-14_20-04-10.json"
    
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract ratings as (theta, sigma) tuples
    calibration_ratings = {}
    final_ratings = results.get("final_ratings", {})
    
    for sample_id, rating_data in final_ratings.items():
        theta = rating_data.get("rating", 0.0)
        sigma = rating_data.get("uncertainty", INITIAL_SIGMA)
        calibration_ratings[sample_id] = (theta, sigma)
    
    print(f"✓ Loaded {len(calibration_ratings)} calibration samples from leaderboard")
    return calibration_ratings


def load_sample_by_id(sample_id: str, dataset_path: str = None) -> Optional[Dict[str, Any]]:
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


def find_k_nearest_neighbors(
    current_theta: float,
    calibration_ratings: Dict[str, Tuple[float, float]],
    k: int
) -> List[Tuple[str, float, float]]:
    """
    Find k samples with ratings closest to the current theta estimate.
    
    Args:
        current_theta: Current rating estimate for the new sample
        calibration_ratings: Dictionary of {sample_id: (theta, sigma)}
        k: Number of nearest neighbors to return
        
    Returns:
        List of (sample_id, theta, sigma) tuples for k nearest neighbors,
        sorted by distance from current_theta
    """
    # Calculate distances and sort
    distances = []
    for sample_id, (theta, sigma) in calibration_ratings.items():
        distance = abs(theta - current_theta)
        distances.append((distance, sample_id, theta, sigma))
    
    # Sort by distance and take top k
    distances.sort(key=lambda x: x[0])
    
    # Return k nearest (without distance in output)
    return [(sample_id, theta, sigma) for _, sample_id, theta, sigma in distances[:k]]


def select_opponent(
    candidates: List[Tuple[str, float, float]],
    current_theta: float
) -> Tuple[str, float, float]:
    """
    Select one opponent from candidates, preferring samples with low sigma
    (high confidence) and theta close to current estimate.
    
    Strategy:
    - Calculate a selection score that balances confidence (1/sigma) and 
      proximity (inverse of distance to current_theta)
    - Select the candidate with highest score
    
    Args:
        candidates: List of (sample_id, theta, sigma) tuples
        current_theta: Current rating estimate
        
    Returns:
        Tuple of (sample_id, theta, sigma) for selected opponent
    """
    if not candidates:
        raise ValueError("Cannot select opponent from empty candidate list")
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Calculate selection scores
    scores = []
    for sample_id, theta, sigma in candidates:
        # Confidence score (prefer low sigma)
        confidence_score = 1.0 / (sigma + 1.0)  # Add 1 to avoid division by zero
        
        # Proximity score (prefer close theta)
        distance = abs(theta - current_theta)
        proximity_score = 1.0 / (distance + 1.0)  # Add 1 to avoid division by zero
        
        # Combined score (weight confidence more heavily)
        # Using 2:1 ratio for confidence:proximity
        combined_score = 2.0 * confidence_score + proximity_score
        
        scores.append((combined_score, sample_id, theta, sigma))
    
    # Select candidate with highest score
    scores.sort(key=lambda x: x[0], reverse=True)
    _, sample_id, theta, sigma = scores[0]
    
    return (sample_id, theta, sigma)


def calculate_rating_from_comparisons(
    new_sample_id: str,
    comparisons: List[Dict[str, Any]],
    calibration_ratings: Dict[str, Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Calculate rating and uncertainty for new sample using Bradley-Terry MM algorithm.
    
    This uses the same approach as the initial leaderboard calculation:
    - Bradley-Terry MM algorithm for ratings  
    - Bayesian posterior variance updates for uncertainty using Fisher information
    - Anchors opponent ratings to their known leaderboard values
    
    Args:
        new_sample_id: ID of the new sample being rated
        comparisons: List of comparison dictionaries with opponent_id and result
        calibration_ratings: Dictionary of {sample_id: (theta, sigma)} for calibration samples
        
    Returns:
        Tuple of (theta, sigma) for the new sample
    """
    if not comparisons:
        # No comparisons yet - return initial values
        return 0.0, INITIAL_SIGMA
    
    # Prepare comparison data for Bradley-Terry algorithm
    # Format: (sample_a, sample_b, result) where result is "A", "B", or "tie"
    bt_comparisons = []
    
    # Get unique opponent IDs
    opponent_ids = set()
    
    for comp in comparisons:
        opponent_id = comp['opponent_id']
        opponent_ids.add(opponent_id)
        result = comp['result']
        
        # Convert result to Bradley-Terry format
        if result == 1.0:
            # New sample wins
            bt_result = "A"
        elif result == 0.0:
            # Opponent wins
            bt_result = "B"
        else:
            # Tie
            bt_result = "tie"
        
        bt_comparisons.append((new_sample_id, opponent_id, bt_result))
    
    # Add anchor comparisons to fix opponent ratings at their leaderboard values
    # Create virtual "anchor" samples for each opponent that always tie with them
    # This constrains the Bradley-Terry algorithm to keep opponents near their known ratings
    for opponent_id in opponent_ids:
        if opponent_id in calibration_ratings:
            # Add multiple anchor ties to strongly constrain this opponent's rating
            # More anchors = stronger constraint
            anchor_id = f"_anchor_{opponent_id}"
            for _ in range(20):  # 20 virtual ties per opponent
                bt_comparisons.append((opponent_id, anchor_id, "tie"))
    
    # Run Bradley-Terry MM on all comparisons (real + anchors)
    all_ratings = bradley_terry_mm(bt_comparisons, max_iterations=500, tolerance=0.01, lambda_reg=0.5)
    
    # Get the rating for the new sample
    theta_raw = all_ratings.get(new_sample_id, 0.0)
    
    # Adjust the rating based on how the opponent ratings shifted
    # Calculate average shift in opponent ratings
    total_shift = 0.0
    for opponent_id in opponent_ids:
        if opponent_id in calibration_ratings:
            opponent_leaderboard_theta, _ = calibration_ratings[opponent_id]
            opponent_bt_theta = all_ratings.get(opponent_id, 0.0)
            shift = opponent_leaderboard_theta - opponent_bt_theta
            total_shift += shift
    
    avg_shift = total_shift / len(opponent_ids) if opponent_ids else 0.0
    
    # Apply correction to put new sample on the same scale as the leaderboard
    theta = theta_raw + avg_shift
    
    # Calculate uncertainty using Bayesian posterior variance updates
    # Start with high initial uncertainty
    sigma = INITIAL_SIGMA
    
    # Update sigma after each comparison using Fisher information
    for comp in comparisons:
        opponent_id = comp['opponent_id']
        if opponent_id in calibration_ratings:
            opponent_theta, _ = calibration_ratings[opponent_id]
            
            # Calculate predicted win probability
            # p_A = 1 / (1 + exp(theta_B - theta_A))
            # Ratings are on log scale (multiplied by 100), so divide by 100
            theta_diff = (opponent_theta - theta) / 100.0
            
            # Use stable computation to avoid overflow
            if theta_diff > 20:
                p_a = 0.0
            elif theta_diff < -20:
                p_a = 1.0
            else:
                p_a = 1.0 / (1.0 + math.exp(theta_diff))
            
            # Calculate Fisher information
            # I = p_A * (1 - p_A)
            # Scale by 100^2 to match the rating scale (ratings are log(strength) * 100)
            fisher_info = p_a * (1.0 - p_a) / (100.0 ** 2)
            
            # Update sigma using Bayesian posterior variance
            # sigma_new = 1 / sqrt(1/sigma^2 + I)
            if fisher_info > 0:
                precision = 1.0 / (sigma ** 2)
                new_precision = precision + fisher_info
                sigma = 1.0 / math.sqrt(new_precision)
    
    return theta, sigma


def run_comparison(
    new_sample: Dict[str, Any],
    opponent_sample: Dict[str, Any],
    api_key: str,
    model: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a pairwise comparison between new sample and opponent.
    
    Args:
        new_sample: The new sample being rated
        opponent_sample: The opponent sample from calibration set
        api_key: OpenRouter API key
        model: OpenRouter model to use
        
    Returns:
        Tuple of (result, full_comparison_data):
        - result: Comparison result (1.0 if new sample wins, 0.0 if loses, 0.5 if tie)
        - full_comparison_data: Full comparison result including arguments and reasoning
    """
    # Extract transcripts
    transcript_new = extract_conversation_to_string(new_sample)
    transcript_opponent = extract_conversation_to_string(opponent_sample)
    
    # Run comparison with explanation enabled
    comparison_result = compare_transcripts_realism(
        transcript_new,
        transcript_opponent,
        api_key,
        model,
        explain_reason=True
    )
    
    # Parse result
    final_winner = comparison_result.get("final_winner", "tie")
    
    if final_winner == "1":
        numeric_result = 1.0  # New sample wins
    elif final_winner == "2":
        numeric_result = 0.0  # Opponent wins
    else:
        numeric_result = 0.5  # Tie
    
    return numeric_result, comparison_result


def integrate_new_sample(
    new_sample: Dict[str, Any],
    calibration_ratings: Dict[str, Tuple[float, float]],
    k_initial: float,
    max_comparisons: int,
    uncertainty_threshold: float,
    api_key: str,
    model: str,
    k_neighbors: int = 10,
    dataset_path: str = None,
    verbose: bool = True,
    comparison_callback: Optional[Callable[[str, int, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Integrate a new sample into the calibration set using adaptive comparisons.
    
    This is the main function that implements the adaptive integration algorithm.
    Uses the same Bradley-Terry MM approach as the initial leaderboard calculation.
    
    Args:
        new_sample: The sample to be rated (in processed dataset format)
        calibration_ratings: Dictionary of {sample_id: (theta, sigma)} for existing samples
        k_initial: Initial update sensitivity (DEPRECATED - not used with Bradley-Terry)
        max_comparisons: Maximum number of comparisons to run
        uncertainty_threshold: Stop when sigma drops below this value
        api_key: OpenRouter API key for running comparisons
        model: OpenRouter model to use for comparisons
        k_neighbors: Number of nearest neighbors to consider for opponent selection
        dataset_path: Path to dataset file (for loading opponent samples)
        verbose: Whether to print progress information
        comparison_callback: Optional callback function(sample_id, comparison_num, comparison_data)
                           called after each comparison completes
        
    Returns:
        Dictionary containing:
        - final_theta: Final rating for the new sample
        - final_sigma: Final uncertainty for the new sample
        - comparisons: List of all comparisons made
        - num_comparisons: Total number of comparisons run
        - convergence_reason: Why the algorithm stopped
    """
    new_sample_id = new_sample.get('id', 'unknown')
    
    # Initialize new sample rating at median of calibration set
    all_thetas = [theta for theta, _ in calibration_ratings.values()]
    all_thetas.sort()
    median_theta = all_thetas[len(all_thetas) // 2] if all_thetas else 0.0
    
    theta_current = median_theta
    sigma_current = INITIAL_SIGMA  # High initial uncertainty
    
    comparisons = []
    
    if verbose:
        print(f"\nStarting adaptive integration for sample: {new_sample_id}")
        print(f"Initial theta estimate: {theta_current:.2f}, sigma: {sigma_current:.2f}")
        print(f"Target uncertainty: {uncertainty_threshold:.2f}")
        print(f"Max comparisons: {max_comparisons}\n")
    
    for comparison_num in range(max_comparisons):
        # Check convergence
        if sigma_current < uncertainty_threshold:
            convergence_reason = f"Uncertainty threshold reached ({sigma_current:.2f} < {uncertainty_threshold:.2f})"
            if verbose:
                print(f"\n✓ Converged: {convergence_reason}")
            break
        
        # Find k nearest neighbors
        neighbors = find_k_nearest_neighbors(
            theta_current,
            calibration_ratings,
            k_neighbors
        )
        
        # Select opponent
        opponent_id, opponent_theta, opponent_sigma = select_opponent(
            neighbors,
            theta_current
        )
        
        if verbose:
            print(f"Comparison {comparison_num + 1}/{max_comparisons}:")
            print(f"  Current estimate: theta={theta_current:.2f}, sigma={sigma_current:.2f}")
            print(f"  Opponent: {opponent_id} (theta={opponent_theta:.2f}, sigma={opponent_sigma:.2f})")
        
        # Load opponent sample
        opponent_sample = load_sample_by_id(opponent_id, dataset_path)
        if opponent_sample is None:
            if verbose:
                print(f"  ⚠ Warning: Could not load opponent sample {opponent_id}, skipping")
            continue
        
        # Run comparison (now returns tuple with result and full comparison data)
        result, full_comparison_data = run_comparison(new_sample, opponent_sample, api_key, model)
        
        if verbose:
            result_str = "WIN" if result == 1.0 else ("LOSS" if result == 0.0 else "TIE")
            print(f"  Result: {result_str} ({result})")
            
            # Log reasoning if available
            individual_results = full_comparison_data.get("individual_results", [])
            if individual_results:
                print(f"  Reasoning from {len(individual_results)} judge comparisons:")
                for idx, indiv_result in enumerate(individual_results, 1):
                    judge_result = indiv_result.get("result", {})
                    arguments = judge_result.get("arguments", [])
                    if arguments:
                        print(f"    Judge {idx} arguments:")
                        for arg in arguments:
                            if isinstance(arg, dict):
                                desc = arg.get("description", "")
                                short = arg.get("short_version", "")
                                print(f"      - {short}: {desc}")
                            else:
                                print(f"      - {arg}")
        
        # Record comparison with full reasoning data
        comparisons.append({
            "comparison_num": comparison_num + 1,
            "opponent_id": opponent_id,
            "opponent_theta": opponent_theta,
            "opponent_sigma": opponent_sigma,
            "result": result,
            "theta_before": theta_current,
            "sigma_before": sigma_current,
            "full_comparison_data": full_comparison_data,  # Store all reasoning
        })
        
        # Recalculate rating using Bradley-Terry on all comparisons so far
        theta_new, sigma_new = calculate_rating_from_comparisons(
            new_sample_id,
            comparisons,
            calibration_ratings
        )
        
        # Update comparison record with new values
        comparisons[-1]["theta_after"] = theta_new
        comparisons[-1]["sigma_after"] = sigma_new
        
        if verbose:
            theta_change = theta_new - theta_current
            sigma_change = sigma_new - sigma_current
            print(f"  Updated: theta={theta_new:.2f} (Δ{theta_change:+.2f}), sigma={sigma_new:.2f} (Δ{sigma_change:+.2f})\n")
        
        # Call callback if provided
        if comparison_callback:
            comparison_callback(new_sample_id, comparison_num + 1, comparisons[-1])
        
        # Update current values
        theta_current = theta_new
        sigma_current = sigma_new
    else:
        # Loop completed without break (max comparisons reached)
        convergence_reason = f"Maximum comparisons reached ({max_comparisons})"
        if verbose:
            print(f"\n→ Stopped: {convergence_reason}")
    
    if verbose:
        print(f"\nFinal rating: theta={theta_current:.2f}, sigma={sigma_current:.2f}")
        print(f"Total comparisons: {len(comparisons)}")
    
    return {
        "new_sample_id": new_sample_id,
        "final_theta": theta_current,
        "final_sigma": sigma_current,
        "initial_theta": median_theta,
        "initial_sigma": INITIAL_SIGMA,
        "comparisons": comparisons,
        "num_comparisons": len(comparisons),
        "convergence_reason": convergence_reason
    }


def calculate_leaderboard_placement(
    theta: float,
    calibration_ratings: Dict[str, Tuple[float, float]]
) -> Tuple[int, int, int]:
    """
    Calculate where a sample would rank on the leaderboard based on its theta.
    
    Args:
        theta: The rating (theta) of the sample
        calibration_ratings: Dictionary of {sample_id: (theta, sigma)} for all calibration samples
        
    Returns:
        Tuple of (rank, total_samples, percentile)
        - rank: Position on leaderboard (1 = highest rating)
        - total_samples: Total number of samples including this one
        - percentile: Percentile ranking (0-100, higher is better)
    """
    # Get all theta values from calibration set
    all_thetas = [t for t, _ in calibration_ratings.values()]
    all_thetas.append(theta)  # Add the new sample
    
    # Sort in descending order (highest rating first)
    all_thetas.sort(reverse=True)
    
    # Find rank (1-indexed)
    rank = all_thetas.index(theta) + 1
    total_samples = len(all_thetas)
    
    # Calculate percentile (higher is better)
    percentile = round(100 * (1 - (rank - 1) / total_samples), 1)
    
    return rank, total_samples, percentile


def save_integration_results(
    results: Dict[str, Any],
    output_dir: str = None,
    verbose: bool = True
) -> str:
    """
    Save integration results to a JSON file with timestamp.
    
    Args:
        results: Integration results dictionary
        output_dir: Directory to save results. If None, uses default 'results' directory.
        verbose: Whether to print confirmation message
        
    Returns:
        Path to saved results file
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "results"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sample_id = results.get("new_sample_id", "unknown")
    # Sanitize sample_id for filename
    sample_id_clean = sample_id.replace(":", "_").replace("/", "_")
    filename = f"integration_{sample_id_clean}_{timestamp}.json"
    
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
    return str(output_path)


def save_aggregate_results(
    all_results: List[Dict[str, Any]],
    output_dir: str = None,
    verbose: bool = True,
    output_name: str = None
) -> str:
    """
    Save all individual integration results into one comprehensive aggregate file.
    
    This includes all detailed comparison data for each sample, not just the summary.
    
    Args:
        all_results: List of all integration results (full detail)
        output_dir: Directory to save results. If None, uses default 'apollo_leaderboard_placement' directory.
        verbose: Whether to print confirmation message
        output_name: Optional base name for output file (without extension)
        
    Returns:
        Path to saved aggregate results file
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "apollo_leaderboard_placement"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp (always needed for metadata)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate filename
    if output_name:
        filename = f"{output_name}.json"
    else:
        filename = f"integration_results_{timestamp}.json"
    output_path = output_dir / filename
    
    # Sort results by sample_id for consistency
    sorted_results = sorted(all_results, key=lambda r: r.get('new_sample_id', ''))
    
    # Create aggregate structure
    aggregate = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples": len(all_results),
            "total_comparisons": sum(r['num_comparisons'] for r in all_results),
            "average_comparisons": sum(r['num_comparisons'] for r in all_results) / len(all_results) if all_results else 0,
            "average_theta": sum(r['final_theta'] for r in all_results) / len(all_results) if all_results else 0,
            "average_sigma": sum(r['final_sigma'] for r in all_results) / len(all_results) if all_results else 0,
        },
        "samples": sorted_results
    }
    
    # Add convergence statistics
    converged = sum(1 for r in all_results if 'threshold' in r.get('convergence_reason', '').lower())
    aggregate["metadata"]["convergence_statistics"] = {
        "reached_threshold": converged,
        "hit_max_comparisons": len(all_results) - converged,
        "convergence_rate": f"{100 * converged / len(all_results):.1f}%" if all_results else "N/A"
    }
    
    # Add leaderboard statistics if available
    if all_results and 'leaderboard_rank' in all_results[0]:
        ranks = [r['leaderboard_rank'] for r in all_results]
        percentiles = [r['leaderboard_percentile'] for r in all_results]
        aggregate["metadata"]["leaderboard_statistics"] = {
            "leaderboard_size": all_results[0].get('leaderboard_total', 0),
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
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Aggregate results saved to: {output_path}")
        print(f"Total samples: {len(all_results)}")
        print(f"Total comparisons: {aggregate['metadata']['total_comparisons']}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"{'='*80}\n")
    
    return str(output_path)


def save_placement_summary(
    all_results: List[Dict[str, Any]],
    output_dir: str = None,
    output_name: str = None
) -> str:
    """
    Save a summary of all leaderboard placements (text format only).
    
    Args:
        all_results: List of all integration results
        output_dir: Directory to save summary. If None, uses default 'apollo_leaderboard_placement' directory.
        output_name: Optional base name for output file (without extension)
        
    Returns:
        Path to saved summary file
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "apollo_leaderboard_placement"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp (always needed for metadata)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate filename
    if output_name:
        txt_filename = f"{output_name}_summary.txt"
    else:
        txt_filename = f"placement_summary_{timestamp}.txt"
    
    txt_path = output_dir / txt_filename
    
    # Sort by rank
    sorted_results = sorted(all_results, key=lambda r: r['leaderboard_rank'])
    
    # Calculate summary statistics
    total_samples = len(all_results)
    total_comparisons = sum(r['num_comparisons'] for r in all_results)
    average_theta = sum(r['final_theta'] for r in all_results) / len(all_results) if all_results else 0
    leaderboard_size = all_results[0]['leaderboard_total'] if all_results else 0
    
    # Save human-readable text version
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LEADERBOARD PLACEMENT SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Total comparisons: {total_comparisons}\n")
        f.write(f"Average theta: {average_theta:.2f}\n")
        f.write(f"Leaderboard size: {leaderboard_size} samples\n")
        f.write("\n" + "="*80 + "\n")
        f.write("PLACEMENTS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Rank':<10} {'Sample ID':<50} {'Rating':<20} {'Percentile':<12}\n")
        f.write("-" * 80 + "\n")
        
        for r in sorted_results:
            rank_str = f"#{r['leaderboard_rank']}/{r['leaderboard_total']}"
            rating_str = f"{r['final_theta']:.2f} ± {r['final_sigma']:.2f}"
            percentile_str = f"{r['leaderboard_percentile']}th"
            f.write(f"{rank_str:<10} {r['new_sample_id']:<50} {rating_str:<20} {percentile_str:<12}\n")
    
    print(f"\n{'='*80}")
    print(f"Placement summary saved to:")
    print(f"  TXT:  {txt_path}")
    print(f"{'='*80}\n")
    
    return str(txt_path)


def main(log_filename: str, leaderboard_path: str, model: str, output_name: str):
    """
    Main function to integrate new samples from an inspect log file into the leaderboard.
    
    Args:
        log_filename: Path to the .eval log file or .json file to process
        leaderboard_path: Path to the Bradley-Terry leaderboard JSON file
        model: OpenRouter model to use for comparisons
        output_name: Base name for output files (without extension), or None for auto-generated
    """
    # Load environment variables from igor_project/.env
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Extract samples from log file
    print(f"\n{'='*80}")
    print("INITIALIZATION")
    print(f"{'='*80}")
    
    # Determine if input is .eval or .json file
    log_path = Path(log_filename)
    if log_path.suffix.lower() == '.json':
        print(f"Loading samples from JSON file...")
        with open(log_path, 'r', encoding='utf-8') as f:
            new_samples = json.load(f)
        print(f"✓ Loaded {len(new_samples)} samples from JSON")
    else:
        print(f"Extracting samples from .eval file...")
        new_samples = extract_prompts_minimal(log_filename)
        print(f"✓ Extracted {len(new_samples)} samples from .eval log")
    
    # Normalize the message key: convert 'prompts' to 'input' if needed
    # (extract_prompts_minimal uses 'prompts', but the rest of the codebase expects 'input')
    for sample in new_samples:
        if 'prompts' in sample and 'input' not in sample:
            sample['input'] = sample.pop('prompts')
    
    # Load calibration ratings
    calibration_ratings = load_calibration_ratings(leaderboard_path)
    
    # Create thread-safe print lock for parallel execution
    print_lock = threading.Lock()
    
    # Create callback for printing individual comparison results
    def comparison_callback(sample_id: str, comparison_num: int, comparison_data: Dict[str, Any]):
        """Thread-safe callback to print each comparison result with reasoning."""
        with print_lock:
            result = comparison_data['result']
            result_str = "WIN" if result == 1.0 else ("LOSS" if result == 0.0 else "TIE")
            opponent_id = comparison_data['opponent_id']
            theta_after = comparison_data['theta_after']
            sigma_after = comparison_data['sigma_after']
            theta_change = theta_after - comparison_data['theta_before']
            
            print(f"\n[{sample_id}] Comparison {comparison_num}: {result_str} vs {opponent_id} | "
                  f"θ={theta_after:.2f} (Δ{theta_change:+.2f}), σ={sigma_after:.2f}")
            
    
    # Define function to process a single sample
    def process_single_sample(sample_index_and_data):
        i, new_sample = sample_index_and_data
        sample_id = new_sample.get('id', 'unknown')
        
        # Run adaptive integration with callback for individual comparison results
        results = integrate_new_sample(
            new_sample=new_sample,
            calibration_ratings=calibration_ratings,
            k_initial=32,
            max_comparisons=1,
            uncertainty_threshold=20.0,
            api_key=api_key,
            model=model,
            k_neighbors=10,
            verbose=False,  # Disable detailed verbose output (using callback instead)
            comparison_callback=comparison_callback
        )
        
        # Calculate leaderboard placement
        rank, total, percentile = calculate_leaderboard_placement(
            results['final_theta'],
            calibration_ratings
        )
        
        # Add placement info to results
        results['leaderboard_rank'] = rank
        results['leaderboard_total'] = total
        results['leaderboard_percentile'] = percentile
        
        return results
    
    # Process all samples in parallel using ThreadPoolExecutor
    print(f"\n{'='*80}")
    print(f"Starting parallel processing of {len(new_samples)} samples...")
    print(f"{'='*80}\n")
    
    all_results = []
    failed_samples = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=len(new_samples)) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_sample, (i, sample)): sample 
            for i, sample in enumerate(new_samples, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed_count += 1
                print(f"✓ Completed {completed_count}/{len(new_samples)}: {result['new_sample_id']} → Rank #{result['leaderboard_rank']}/{result['leaderboard_total']} ({result['leaderboard_percentile']}th percentile)")
            except Exception as e:
                sample = futures[future]
                completed_count += 1
                sample_id = sample.get('id', 'unknown')
                failed_samples.append(sample_id)
                
                print(f"\n{'!'*80}")
                print(f"✗ FAILED {completed_count}/{len(new_samples)}: {sample_id}")
                print(f"Error: {type(e).__name__}: {e}")
                print(f"{'!'*80}")
                print("Traceback:")
                traceback.print_exc()
                print(f"{'!'*80}\n")
    
    # Print overall summary with leaderboard placements
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total samples attempted: {len(new_samples)}")
    print(f"Successfully processed: {len(all_results)}")
    if failed_samples:
        print(f"Failed: {len(failed_samples)}")
        print(f"Failed sample IDs: {', '.join(failed_samples)}")
    
    if all_results:
        print(f"\nTotal comparisons: {sum(r['num_comparisons'] for r in all_results)}")
        avg_comparisons = sum(r['num_comparisons'] for r in all_results) / len(all_results)
        print(f"Average comparisons per sample: {avg_comparisons:.1f}")
        avg_theta = sum(r['final_theta'] for r in all_results) / len(all_results)
        avg_sigma = sum(r['final_sigma'] for r in all_results) / len(all_results)
        print(f"Average final theta: {avg_theta:.2f}")
        print(f"Average final uncertainty (sigma): {avg_sigma:.2f}")
        
        # Convergence statistics
        converged = sum(1 for r in all_results if 'threshold' in r['convergence_reason'].lower())
        print(f"\nConvergence statistics:")
        print(f"  Reached uncertainty threshold: {converged}/{len(all_results)}")
        print(f"  Hit max comparisons: {len(all_results) - converged}/{len(all_results)}")
    else:
        print("\n⚠ No samples were successfully processed!")
    
    # Only show detailed results if there are successful samples
    if all_results:
        print("\n" + "="*80)
        print("DETAILED LEADERBOARD PLACEMENTS")
        print("="*80)
        print(f"{'Rank':<10} {'Sample ID':<50} {'Rating':<20} {'Percentile':<12} {'Comparisons':<12}")
        print("-" * 80)
        
        # Sort results by rank
        sorted_results = sorted(all_results, key=lambda r: r['leaderboard_rank'])
        
        for r in sorted_results:
            sample_id = r['new_sample_id']
            rating = f"{r['final_theta']:.2f} ± {r['final_sigma']:.2f}"
            rank = f"#{r['leaderboard_rank']}/{r['leaderboard_total']}"
            percentile = f"{r['leaderboard_percentile']}th"
            comparisons = str(r['num_comparisons'])
            print(f"{rank:<10} {sample_id:<50} {rating:<20} {percentile:<12} {comparisons:<12}")
        
        # Additional statistics
        print("\n" + "="*80)
        print("PLACEMENT DISTRIBUTION")
        print("="*80)
        ranks = [r['leaderboard_rank'] for r in all_results]
        percentiles = [r['leaderboard_percentile'] for r in all_results]
        print(f"Best placement: #{min(ranks)} ({max(percentiles)}th percentile)")
        print(f"Worst placement: #{max(ranks)} ({min(percentiles)}th percentile)")
        print(f"Median placement: #{sorted(ranks)[len(ranks)//2]} ({sorted(percentiles)[len(percentiles)//2]}th percentile)")
        
        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save comprehensive aggregate results with all comparison details
        save_aggregate_results(all_results, output_name=output_name)
        
        # Save placement summary
        save_placement_summary(all_results, output_name=output_name)
    else:
        print("\n⚠ Skipping detailed results and file saving due to no successful samples.")


def cli_main():
    """
    CLI entry point that parses command line arguments and calls main().
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Integrate new samples from an eval log or JSON file into the Bradley-Terry leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with .eval file (model is required)
  python sample_integration.py 2025-10-19T14-32-43+03-00_oversight-subversion_2s8d5NcbFfm9ffNrzJZyjt.eval --model openai/gpt-5-mini
  python sample_integration.py mylog.eval -m anthropic/claude-sonnet-4.5
  
  # Using JSON file directly (skips .eval to JSON conversion)
  python sample_integration.py partial_transcripts/partial_transcripts_swe_gym_4458.json --model x-ai/grok-4-fast
  python sample_integration.py mysamples.json -m openai/gpt-5-mini
  
  # Using full path to eval or JSON file
  python sample_integration.py /path/to/logfile.eval --model openai/gpt-5-mini
  python sample_integration.py /path/to/samples.json --model openai/gpt-5-mini
  
  # Specifying custom leaderboard file
  python sample_integration.py mylog.eval --model openai/gpt-5-mini --leaderboard custom_leaderboard.json
  
  # Specifying custom output name (creates my_results.json and my_results_summary.txt)
  python sample_integration.py mylog.eval -m openai/gpt-5-mini --output my_results
  python sample_integration.py samples.json -m openai/gpt-5-mini -o my_results
  
Note: Results are saved to the 'apollo_leaderboard_placement' directory
      JSON files should contain a list of samples with 'id', 'prompts', and 'metadata' fields
        """
    )
    
    parser.add_argument(
        'eval_file',
        type=str,
        help='Name or path of the .eval log file or .json file to process. If just a filename, looks in eval_logs/ or partial_transcripts/'
    )
    
    parser.add_argument(
        '--leaderboard',
        type=str,
        default=None,
        help='Path to the Bradley-Terry leaderboard JSON file'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Base name for output files (default: auto-generated with timestamp). Extension .json will be added.'
    )
    
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        required=True,
        help='OpenRouter model to use for comparisons (e.g., openai/gpt-5-mini, anthropic/claude-sonnet-4.5)'
    )
    
    args = parser.parse_args()
    
    log_file_input = args.eval_file
    
    log_file_path = Path(log_file_input)
    
    # If it's not an absolute path and doesn't exist, try looking in different directories
    if not log_file_path.is_absolute() and not log_file_path.exists():
        script_dir = Path(__file__).parent
        
        # Try eval_logs/ directory for .eval files
        potential_eval_path = script_dir / "eval_logs" / log_file_input
        
        # Try partial_transcripts/ directory for .json files
        potential_json_path = script_dir / "partial_transcripts" / log_file_input
        
        if potential_eval_path.exists():
            log_file_path = potential_eval_path
        elif potential_json_path.exists():
            log_file_path = potential_json_path
        else:
            print(f"Error: Could not find file at {log_file_path}")
            print(f"  Also checked: {potential_eval_path}")
            print(f"  Also checked: {potential_json_path}")
            sys.exit(1)
    
    # Check if log file exists
    if not log_file_path.exists():
        print(f"Error: File not found: {log_file_path}")
        sys.exit(1)
    
    # Determine leaderboard file path
    if args.leaderboard:
        leaderboard_file = Path(args.leaderboard)
    
    # Check if leaderboard file exists
    if not leaderboard_file.exists():
        print(f"Error: Leaderboard file not found: {leaderboard_file}")
        sys.exit(1)
    
    # Determine file type for display
    file_type = "JSON file" if log_file_path.suffix.lower() == '.json' else ".eval file"
    print(f"Using input file: {log_file_path} ({file_type})")
    print(f"Using leaderboard: {leaderboard_file}")
    print(f"Using model: {args.model}")
    if args.output:
        print(f"Output base name: {args.output}")
    
    main(log_file_path, leaderboard_file, args.model, args.output)


if __name__ == "__main__":
    # Check if script is being run with CLI arguments
    if len(sys.argv) > 1:
        # CLI mode - parse arguments and run
        cli_main()
    else:
        # Direct execution mode - run with manually defined parameters
        script_dir = Path(__file__).parent
        
        # Define parameters for direct execution
        log_file = script_dir / "eval_logs" / "2025-10-19T14-32-43+03-00_oversight-subversion_2s8d5NcbFfm9ffNrzJZyjt.eval"
        leaderboard_file = script_dir / "leaderboards" / "grok_4_fast_leaderboard.json"
        model = "x-ai/grok-4-fast"
        output_name = "explanations_test"  
        print("="*80)
        print("RUNNING IN DIRECT EXECUTION MODE")
        print("="*80)
        
        # Determine file type
        file_type = "JSON file" if Path(log_file).suffix.lower() == '.json' else ".eval file"
        print(f"Using input file: {log_file} ({file_type})")
        print(f"Using leaderboard: {leaderboard_file}")
        print(f"Using model: {model}")
        print("="*80 + "\n")
        
        main(log_file, leaderboard_file, model, output_name)


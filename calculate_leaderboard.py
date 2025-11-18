"""
Calculate Bradley-Terry leaderboard from pairwise comparisons of conversation transcripts.

This module implements a complete leaderboard calculation system using the Bradley-Terry
model to rank conversation transcripts by their realism. It uses parallel processing to compare samples.

Main features:
- Bradley-Terry MM algorithm for rating calculation
- Adaptive pairing strategy (exploration then exploitation)
- Parallel comparison execution for speed
- Detailed logging of all judge decisions
"""

import json
import random
import os
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dotenv import load_dotenv
from sample_comparison import extract_conversation_to_string, compare_transcripts_realism
from bradley_terry import (
    bradley_terry_mm,
    calculate_uncertainties,
    get_comparison_counts,
    get_rating_statistics,
    get_uncertainty_statistics,
    INITIAL_SIGMA
)
from sanity_checks.basic_sanity_checks import run_all_sanity_checks


AGENTIC_GROUPS = {
    'aider_commit', 'aider_manual', 'aider_multi', 'continue_dev', 'irl_cursor', 'openhands',
    'smol_real_manager', 'smol_real_search', 'agentharm', 'aider_full', 'cybench', 'gaia_runs',
    'honeypot', 'mvp', 'smol_gaia_manager', 'smol_gaia_search', 'swe_gym', 'swebench'
}


def load_dataset(dataset_path: str = None) -> List[Dict[str, Any]]:
    """
    Load all samples from the processed Needham dataset.
    
    Args:
        dataset_path: Path to the dataset file. If None, uses default path.
        
    Returns:
        List of sample dictionaries
    """
    if dataset_path is None:
        # Default path relative to this script
        script_dir = Path(__file__).parent
        dataset_path = script_dir / "datasets" / "processed_cleaned_needham_dataset.json"
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples from dataset")
    return samples


def compare_random_samples(samples: List[Dict[str, Any]], api_key: str, model: str, agentic_only: bool = False) -> Dict[str, Any]:
    """
    Randomly select 2 samples from the dataset and compare their realism.
    
    Args:
        samples: List of sample dictionaries from the dataset
        api_key: OpenRouter API key
        model: OpenRouter model to use
        agentic_only: If True, only compare samples from agentic categories
        
    Returns:
        Dictionary containing comparison result
    """
    # Filter for agentic samples if requested
    if agentic_only:
        samples = [s for s in samples if s.get('metadata', {}).get('category') in AGENTIC_GROUPS]
    
    # Randomly select 2 different samples
    sample1, sample2 = random.sample(samples, 2)
    
    # Extract conversations to strings
    transcript1 = extract_conversation_to_string(sample1)
    transcript2 = extract_conversation_to_string(sample2)


    # Compare using LLM
    comparison_result = compare_transcripts_realism(transcript1, transcript2, api_key, model)
    
    return {
        "sample1_id": sample1.get('id'),
        "sample2_id": sample2.get('id'),
        "comparison_result": comparison_result
    }


def generate_random_pairs(samples: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Generate random pairs for exploration rounds (1-3).
    
    Args:
        samples: List of samples
    
    Returns:
        List of (index_i, index_j) pairs
    """
    n = len(samples)
    indices = list(range(n))
    random.shuffle(indices)
    
    pairs = []
    # Pair consecutive samples in the shuffled list
    for i in range(0, n - 1, 2):
        pairs.append((indices[i], indices[i + 1]))
    
    return pairs


def generate_similarity_pairs(
    samples: List[Dict[str, Any]],
    ratings: Dict[str, float],
    uncertainties: Dict[str, float],
    comparison_history: set
) -> List[Tuple[int, int]]:
    """
    Generate pairs based on similarity in ratings (exploitation rounds 4+).
    Priority: highest uncertainty samples paired with most similar ratings.
    
    Args:
        samples: List of samples
        ratings: Current ratings dictionary
        uncertainties: Current uncertainties dictionary
        comparison_history: Set of previously compared pairs
    
    Returns:
        List of (index_i, index_j) pairs
    """
    # Create list of (index, sample_id, rating, uncertainty)
    sample_info = []
    for i, sample in enumerate(samples):
        sample_id = sample['id']
        rating = ratings.get(sample_id, 0.0)
        uncertainty = uncertainties.get(sample_id, INITIAL_SIGMA)
        sample_info.append((i, sample_id, rating, uncertainty))
    
    # Sort by uncertainty (highest first)
    sample_info.sort(key=lambda x: x[3], reverse=True)
    
    pairs = []
    paired_indices = set()
    
    for idx, sample_id, rating, uncertainty in sample_info:
        if idx in paired_indices:
            continue
        
        # Find best match: closest rating that hasn't been compared yet
        best_match = None
        best_distance = float('inf')
        
        for j, other_id, other_rating, _ in sample_info:
            if j == idx or j in paired_indices:
                continue
            
            # Check if already compared
            pair_key = tuple(sorted([sample_id, other_id]))
            if pair_key in comparison_history:
                continue
            
            # Calculate rating distance
            distance = abs(rating - other_rating)
            if distance < best_distance:
                best_distance = distance
                best_match = j
        
        if best_match is not None:
            pairs.append((idx, best_match))
            paired_indices.add(idx)
            paired_indices.add(best_match)
    
    return pairs


def generate_pairs(
    samples: List[Dict[str, Any]],
    round_num: int,
    ratings: Dict[str, float],
    uncertainties: Dict[str, float],
    comparison_history: set
) -> List[Tuple[int, int]]:
    """
    Generate pairs for comparison based on round number.
    
    Rounds 1-3: Random pairing (exploration)
    Rounds 4+: Similarity-based pairing (exploitation)
    
    Args:
        samples: List of samples
        round_num: Current round number (1-indexed)
        ratings: Current ratings dictionary
        uncertainties: Current uncertainties dictionary
        comparison_history: Set of previously compared pairs
    
    Returns:
        List of (index_i, index_j) pairs
    """
    if round_num <= 3:
        return generate_random_pairs(samples)
    else:
        return generate_similarity_pairs(samples, ratings, uncertainties, comparison_history)


def perform_comparison(
    sample1: Dict[str, Any],
    sample2: Dict[str, Any],
    api_key: str,
    model: str,
    return_details: bool = False
) -> Any:
    """
    Compare two samples and return result.
    
    Uses the updated compare_transcripts_realism which performs multiple comparisons
    with 5 different prompts (each in both orders) and aggregates the results.
    
    Args:
        sample1: First sample
        sample2: Second sample
        api_key: OpenRouter API key
        model: Model to use
        return_details: If True, return tuple of (result, full_comparison_data)
    
    Returns:
        If return_details is False: Result string ("A", "B", "tie", or "error")
        If return_details is True: Tuple of (result, full_comparison_data) where
            full_comparison_data contains all individual judge decisions and reasoning.
            "error" is returned when all individual comparisons fail or have invalid data.
    """
    transcript1 = extract_conversation_to_string(sample1)
    transcript2 = extract_conversation_to_string(sample2)
    
    comparison_result = compare_transcripts_realism(transcript1, transcript2, api_key, model)
    
    # Parse result from the new format (uses "final_winner" instead of "more_realistic")
    final_winner = comparison_result.get("final_winner", "tie").lower()
    
    if final_winner == "1":
        result = "A"
    elif final_winner == "2":
        result = "B"
    elif final_winner == "error":
        result = "error"
    else:
        result = "tie"
    
    if return_details:
        return result, comparison_result
    else:
        return result


def compare_pair_wrapper(
    idx1: int,
    idx2: int,
    samples: List[Dict[str, Any]],
    api_key: str,
    model: str,
    comparison_history: set
) -> Tuple[str, str, str, Optional[str], int, int, Optional[Dict[str, Any]]]:
    """
    Wrapper function for parallel comparison execution.
    
    Args:
        idx1: Index of first sample
        idx2: Index of second sample
        samples: List of all samples
        api_key: OpenRouter API key
        model: Model to use
        comparison_history: Set of already compared pairs
    
    Returns:
        Tuple of (sample1_id, sample2_id, result, error, idx1, idx2, detailed_comparison_data)
        where result can be "A", "B", "tie", "error", or "skip"
    """
    sample1 = samples[idx1]
    sample2 = samples[idx2]
    
    # Check if already compared
    pair_key = tuple(sorted([sample1['id'], sample2['id']]))
    if pair_key in comparison_history:
        return sample1['id'], sample2['id'], "skip", "Duplicate comparison", idx1, idx2, None
    
    try:
        result, detailed_data = perform_comparison(sample1, sample2, api_key, model, return_details=True)
        return sample1['id'], sample2['id'], result, None, idx1, idx2, detailed_data
    except Exception as e:
        return sample1['id'], sample2['id'], "error", str(e), idx1, idx2, None


def run_bradley_terry_competition(
    samples: List[Dict[str, Any]],
    api_key: str,
    model: str,
    max_rounds,
    max_uncertainty_threshold,
    output_dir: Optional[str] = None,
    max_workers: int = 10,
    comparison_timeout: float = 300.0
) -> Dict[str, Any]:
    """
    Run a Bradley-Terry competition to rate samples by realism with parallel comparisons.
    
    Each comparison uses 5 different judge prompts (each applied in both orders) and
    aggregates the results. All individual judge decisions and reasoning are logged.
    
    Comparisons that encounter API errors, timeouts, or have no valid votes are discarded
    and not included in the rating calculations.
    
    Args:
        samples: List of samples to compare
        api_key: OpenRouter API key
        model: Model to use for comparisons
        max_rounds: Maximum number of comparison rounds
        max_uncertainty_threshold: Stop if max uncertainty drops below this
        output_dir: Directory to save results (None = don't save)
        max_workers: Maximum number of parallel workers for comparisons
        comparison_timeout: Maximum time (in seconds) to wait for a single comparison (default: 300)
    
    Returns:
        Dictionary with final ratings, uncertainties, comparison history (simple),
        detailed_comparisons (with all judge decisions and reasoning), and diagnostics
    """
    print(f"\n{'='*70}")
    print(f"Starting Bradley-Terry Competition")
    print(f"{'='*70}")
    print(f"Samples: {len(samples)}")
    print(f"Max rounds: {max_rounds}")
    print(f"Uncertainty threshold: {max_uncertainty_threshold}")
    print(f"Max parallel workers: {max_workers}")
    print(f"Comparison timeout: {comparison_timeout}s")
    print(f"{'='*70}\n")
    
    # Initialize
    ratings = {sample['id']: 0.0 for sample in samples}
    uncertainties = {sample['id']: INITIAL_SIGMA for sample in samples}
    comparison_history = set()
    all_comparisons = []
    detailed_comparisons = []  # Store detailed judge decisions and reasoning
    round_diagnostics = []
    
    # Run rounds
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        round_start_time = time.time()
        
        # Generate pairs
        pairs = generate_pairs(samples, round_num, ratings, uncertainties, comparison_history)
        print(f"Generated {len(pairs)} pairs")
        
        if not pairs:
            print("No more pairs to compare!")
            break
        
        print(f"Running {len(pairs)} comparisons in parallel...")
        
        # Perform comparisons in parallel
        round_comparisons = []
        completed_count = 0
        skipped_count = 0
        error_count = 0
        timeout_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all comparison tasks
            future_to_pair = {
                executor.submit(
                    compare_pair_wrapper,
                    idx1, idx2, samples, api_key, model, comparison_history
                ): (idx1, idx2)
                for idx1, idx2 in pairs
            }
            
            # Process results as they complete
            for future in as_completed(future_to_pair):
                try:
                    # Wait for result with timeout
                    sample1_id, sample2_id, result, error, idx1, idx2, detailed_data = future.result(timeout=comparison_timeout)
                    
                    if result == "skip":
                        skipped_count += 1
                        print(f"  ⊘ Skipped duplicate: {sample1_id} vs {sample2_id}")
                        continue
                    
                    # Skip comparisons that had errors or no valid votes
                    if result == "error":
                        error_count += 1
                        skipped_count += 1
                        if error:
                            print(f"  ✗ {sample1_id} vs {sample2_id}: Error - {error[:50]}... (discarded)")
                        else:
                            print(f"  ✗ {sample1_id} vs {sample2_id}: No valid votes (discarded)")
                        continue
                    
                    completed_count += 1
                    
                    # Show final result and vote counts from all judges
                    if detailed_data:
                        vote_counts = detailed_data.get('vote_counts', {})
                        print(f"  ✓ {sample1_id} vs {sample2_id}: {result} (votes: 1={vote_counts.get('1', 0)}, 2={vote_counts.get('2', 0)}, tie={vote_counts.get('tie', 0)})")
                    else:
                        print(f"  ✓ {sample1_id} vs {sample2_id}: {result}")
                    
                    # Record comparison
                    comparison = (sample1_id, sample2_id, result)
                    round_comparisons.append(comparison)
                    all_comparisons.append(comparison)
                    
                    # Record detailed comparison data with all judge decisions
                    if detailed_data:
                        detailed_comparison = {
                            "sample1_id": sample1_id,
                            "sample2_id": sample2_id,
                            "final_result": result,
                            "round": round_num,
                            "vote_counts": detailed_data.get('vote_counts', {}),
                            "individual_results": detailed_data.get('individual_results', []),
                            "error": error
                        }
                        detailed_comparisons.append(detailed_comparison)
                    
                    # Add to history
                    pair_key = tuple(sorted([sample1_id, sample2_id]))
                    comparison_history.add(pair_key)
                
                except TimeoutError:
                    # Handle timeout - discard this comparison entirely
                    idx1, idx2 = future_to_pair[future]
                    sample1_id = samples[idx1]['id']
                    sample2_id = samples[idx2]['id']
                    
                    timeout_count += 1
                    skipped_count += 1
                    
                    print(f"  ⏱ {sample1_id} vs {sample2_id}: TIMEOUT after {comparison_timeout}s (discarded)")
                    
                    # Do NOT record this comparison or add to history
                    # This allows the pair to potentially be compared again in future rounds
                    
                    # Cancel the future to free resources
                    future.cancel()
        
        # Update ratings using Bradley-Terry MM
        print(f"\nUpdating ratings...")
        ratings = bradley_terry_mm(all_comparisons)
        uncertainties = calculate_uncertainties(ratings, all_comparisons)
        
        # Calculate diagnostics
        max_uncertainty = max(uncertainties.values()) if uncertainties else 0
        mean_uncertainty = sum(uncertainties.values()) / len(uncertainties) if uncertainties else 0
        
        round_time = time.time() - round_start_time
        
        diagnostics = {
            "round": round_num,
            "pairs_compared": completed_count,
            "pairs_skipped": skipped_count,
            "pairs_with_errors": error_count,
            "pairs_timed_out": timeout_count,
            "max_uncertainty": max_uncertainty,
            "mean_uncertainty": mean_uncertainty,
            "round_time_seconds": round_time
        }
        round_diagnostics.append(diagnostics)
        
        print(f"\nRound {round_num} complete:")
        print(f"  Comparisons: {completed_count} completed, {skipped_count} skipped, {error_count} errors")
        if timeout_count > 0:
            print(f"  Timeouts: {timeout_count} comparisons timed out")
        print(f"  Max uncertainty: {max_uncertainty:.2f}")
        print(f"  Mean uncertainty: {mean_uncertainty:.2f}")
        print(f"  Time: {round_time:.1f}s")
        if completed_count > 0:
            print(f"  Speed: {completed_count/round_time:.2f} comparisons/second")
        
        # Check stopping criteria
        if max_uncertainty < max_uncertainty_threshold:
            print(f"\n✓ Stopping: Max uncertainty ({max_uncertainty:.2f}) below threshold ({max_uncertainty_threshold})")
            break
    
    # Final results
    print(f"\n{'='*70}")
    print(f"Competition Complete")
    print(f"{'='*70}")
    print(f"Total rounds: {len(round_diagnostics)}")
    print(f"Total comparisons: {len(all_comparisons)}")
    print(f"Detailed judge decisions logged: {len(detailed_comparisons)}")
    print(f"{'='*70}\n")
    
    # Prepare final ratings dictionary
    comparison_counts = get_comparison_counts(all_comparisons)
    final_ratings = {}
    for sample_id in ratings.keys():
        final_ratings[sample_id] = {
            "rating": ratings[sample_id],
            "uncertainty": uncertainties[sample_id],
            "n_comparisons": comparison_counts.get(sample_id, 0)
        }
    
    # Calculate statistics
    rating_stats = get_rating_statistics(ratings)
    uncertainty_stats = get_uncertainty_statistics(uncertainties)
    
    # Prepare results
    results = {
        "final_ratings": final_ratings,
        "comparison_history": [
            {"sample_a": a, "sample_b": b, "result": r}
            for a, b, r in all_comparisons
        ],
        "detailed_comparisons": detailed_comparisons,  # All judge decisions and reasoning
        "round_diagnostics": round_diagnostics,
        "summary_statistics": {
            "ratings": rating_stats,
            "uncertainties": uncertainty_stats,
            "total_comparisons": len(all_comparisons),
            "total_rounds": len(round_diagnostics)
        },
        "metadata": {
            "model": model,
            "max_rounds": max_rounds,
            "max_uncertainty_threshold": max_uncertainty_threshold,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Save to file if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"grok_4_fast.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")
    
    # Run sanity checks
    run_all_sanity_checks(ratings, uncertainties, all_comparisons)
    
    return results


def main():
    # Load environment variables from igor_project/.env
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv('OPENROUTER_API_KEY')
    model = "x-ai/grok-4-fast"
    
    # Load dataset
    samples = load_dataset()
    
    # Filter for agentic samples only
    agentic_samples = [s for s in samples if s.get('metadata', {}).get('category') in AGENTIC_GROUPS]
    
    print(f"Total samples: {len(samples)}")
    print(f"Agentic samples: {len(agentic_samples)}")
    
    # Use absolute path for results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "leaderboards"
    
    # Run competition with parallel comparisons
    results = run_bradley_terry_competition(
        samples=agentic_samples,
        api_key=api_key,
        model=model,
        max_rounds=15,
        max_uncertainty_threshold=20.0,
        output_dir=str(results_dir),  # Use absolute path
        max_workers=600,
        comparison_timeout=750.0
    )
    
    # Print top 5 rated samples
    print("\n" + "="*70)
    print("Top 5 Most Realistic Samples")
    print("="*70)
    sorted_ratings = sorted(
        results["final_ratings"].items(),
        key=lambda x: x[1]["rating"],
        reverse=True
    )
    for i, (sample_id, info) in enumerate(sorted_ratings[:5], 1):
        print(f"{i}. {sample_id}")
        print(f"   Rating: {info['rating']:.2f} ± {info['uncertainty']:.2f}")
        print(f"   Comparisons: {info['n_comparisons']}")
        print()


if __name__ == "__main__":
    main()


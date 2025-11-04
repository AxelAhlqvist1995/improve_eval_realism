"""
Sanity checks and validation functions for the Bradley-Terry rating system.

This module provides functions to validate the correctness of the rating system:
- Verify rating normalization
- Check convergence behavior
- Validate comparison distributions
"""

from typing import Dict, List, Tuple
import math


def check_rating_normalization(ratings: Dict[str, float], tolerance: float = 1.0) -> bool:
    """
    Verify that ratings are properly centered around 0.
    
    Args:
        ratings: Dictionary of sample ratings
        tolerance: Acceptable deviation from 0 mean
    
    Returns:
        True if ratings are properly normalized, False otherwise
    """
    if not ratings:
        return True
    
    mean_rating = sum(ratings.values()) / len(ratings)
    is_normalized = abs(mean_rating) < tolerance
    
    if not is_normalized:
        print(f"❌ Rating normalization check failed: mean = {mean_rating:.4f} (tolerance: {tolerance})")
    else:
        print(f"✓ Rating normalization check passed: mean = {mean_rating:.4f}")
    
    return is_normalized


def check_high_uncertainty_samples(
    uncertainties: Dict[str, float],
    comparisons: List[Tuple[str, str, str]],
    min_comparisons: int = 3
) -> bool:
    """
    Check that high-uncertainty samples have fewer comparisons.
    
    Args:
        uncertainties: Dictionary of sample uncertainties
        comparisons: List of comparisons made
        min_comparisons: Minimum expected comparisons per sample
    
    Returns:
        True if high-uncertainty samples have appropriate comparison counts
    """
    from bradley_terry import get_comparison_counts
    
    comparison_counts = get_comparison_counts(comparisons)
    
    # Find samples with high uncertainty
    mean_uncertainty = sum(uncertainties.values()) / len(uncertainties) if uncertainties else 0
    high_uncertainty_samples = [
        sample for sample, unc in uncertainties.items()
        if unc > mean_uncertainty
    ]
    
    # Check if they have fewer comparisons
    issues = []
    for sample in high_uncertainty_samples:
        count = comparison_counts.get(sample, 0)
        if count < min_comparisons:
            issues.append(f"Sample {sample}: {count} comparisons (uncertainty: {uncertainties[sample]:.2f})")
    
    if issues:
        print(f"⚠ High-uncertainty samples with few comparisons:")
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
        return False
    else:
        print(f"✓ High-uncertainty check passed: samples have appropriate comparison counts")
        return True


def check_mm_convergence(
    comparisons: List[Tuple[str, str, str]],
    max_iterations: int = 500,
    tolerance: float = 0.01,
    lambda_reg: float = 0.5
) -> Dict[str, any]:
    """
    Check if the MM algorithm converges and report convergence statistics.
    
    Args:
        comparisons: List of comparisons
        max_iterations: Maximum iterations to test
        tolerance: Convergence tolerance
        lambda_reg: Regularization parameter (should match bradley_terry.py default)
    
    Returns:
        Dictionary with convergence information
    """
    from bradley_terry import bradley_terry_mm
    
    if not comparisons:
        return {"converged": True, "iterations": 0, "message": "No comparisons to process"}
    
    # Run MM with iteration tracking
    all_samples = set()
    for sample_i, sample_j, _ in comparisons:
        all_samples.add(sample_i)
        all_samples.add(sample_j)
    all_samples = list(all_samples)
    
    # Initialize ratings
    ratings = {sample_id: 1.0 for sample_id in all_samples}
    
    converged = False
    final_iteration = 0
    
    for iteration in range(max_iterations):
        old_ratings = ratings.copy()
        
        # Update step for each sample
        for sample_i in all_samples:
            wins_i = sum(
                1 for (a, b, result) in comparisons
                if (a == sample_i and result == "A") or (b == sample_i and result == "B")
            )
            ties_i = sum(
                1 for (a, b, result) in comparisons
                if (a == sample_i or b == sample_i) and result == "tie"
            )
            denominator = lambda_reg
            for sample_a, sample_b, _ in comparisons:
                if sample_a == sample_i:
                    denominator += 1.0 / (old_ratings[sample_i] + old_ratings[sample_b])
                elif sample_b == sample_i:
                    denominator += 1.0 / (old_ratings[sample_i] + old_ratings[sample_a])
            
            if denominator > 0:
                ratings[sample_i] = (wins_i + 0.5 * ties_i + lambda_reg) / denominator
            else:
                ratings[sample_i] = 1.0
        
        # Normalize
        mean_rating = sum(ratings.values()) / len(ratings) if ratings else 1.0
        if mean_rating > 0:
            # Normalize old_ratings too for fair comparison
            old_mean = sum(old_ratings.values()) / len(old_ratings) if old_ratings else 1.0
            old_ratings_normalized = {k: v / old_mean for k, v in old_ratings.items()}
            ratings = {k: v / mean_rating for k, v in ratings.items()}
        else:
            old_ratings_normalized = old_ratings
        
        # Check convergence (compare normalized values)
        max_change = max(abs(ratings[k] - old_ratings_normalized[k]) for k in ratings) if ratings else 0
        if max_change < tolerance:
            converged = True
            final_iteration = iteration + 1
            break
        
        final_iteration = iteration + 1
    
    result = {
        "converged": converged,
        "iterations": final_iteration,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "lambda_reg": lambda_reg
    }
    
    if converged:
        print(f"✓ MM algorithm converged in {final_iteration} iterations (tolerance: {tolerance}, λ_reg: {lambda_reg})")
    else:
        print(f"❌ MM algorithm did not converge within {max_iterations} iterations (λ_reg: {lambda_reg})")
    
    return result


def check_no_duplicate_comparisons(
    comparisons: List[Tuple[str, str, str]]
) -> bool:
    """
    Check that there are no duplicate comparisons.
    
    Args:
        comparisons: List of comparisons
    
    Returns:
        True if no duplicates found, False otherwise
    """
    seen_pairs = set()
    duplicates = []
    
    for sample_a, sample_b, result in comparisons:
        # Normalize pair order (smaller ID first)
        pair = tuple(sorted([sample_a, sample_b]))
        
        if pair in seen_pairs:
            duplicates.append(pair)
        else:
            seen_pairs.add(pair)
    
    if duplicates:
        print(f"❌ Found {len(duplicates)} duplicate comparisons")
        for dup in duplicates[:5]:
            print(f"  - {dup}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")
        return False
    else:
        print(f"✓ No duplicate comparisons found")
        return True


def check_comparison_distribution(
    comparisons: List[Tuple[str, str, str]]
) -> Dict[str, int]:
    """
    Analyze the distribution of comparison results.
    
    Args:
        comparisons: List of comparisons
    
    Returns:
        Dictionary with result counts
    """
    result_counts = {"A": 0, "B": 0, "tie": 0, "error": 0}
    
    for _, _, result in comparisons:
        result_lower = result.lower() if isinstance(result, str) else "error"
        if result_lower == "a" or result_lower == "A":
            result_counts["A"] += 1
        elif result_lower == "b" or result_lower == "B":
            result_counts["B"] += 1
        elif result_lower == "tie":
            result_counts["tie"] += 1
        else:
            result_counts["error"] += 1
    
    total = len(comparisons)
    print(f"\n📊 Comparison result distribution (total: {total}):")
    print(f"  - A wins: {result_counts['A']} ({result_counts['A']/total*100:.1f}%)")
    print(f"  - B wins: {result_counts['B']} ({result_counts['B']/total*100:.1f}%)")
    print(f"  - Ties: {result_counts['tie']} ({result_counts['tie']/total*100:.1f}%)")
    if result_counts['error'] > 0:
        print(f"  - Errors: {result_counts['error']} ({result_counts['error']/total*100:.1f}%)")
    
    return result_counts


def run_all_sanity_checks(
    ratings: Dict[str, float],
    uncertainties: Dict[str, float],
    comparisons: List[Tuple[str, str, str]]
) -> Dict[str, bool]:
    """
    Run all sanity checks and return results.
    
    Args:
        ratings: Dictionary of sample ratings
        uncertainties: Dictionary of sample uncertainties
        comparisons: List of comparisons made
    
    Returns:
        Dictionary mapping check names to pass/fail status
    """
    print("\n" + "="*60)
    print("Running Sanity Checks")
    print("="*60 + "\n")
    
    results = {}
    
    # Check 1: Rating normalization
    results["normalization"] = check_rating_normalization(ratings)
    print()
    
    # Check 2: No duplicate comparisons
    results["no_duplicates"] = check_no_duplicate_comparisons(comparisons)
    print()
    
    # Check 3: MM convergence
    convergence_info = check_mm_convergence(comparisons)
    results["convergence"] = convergence_info["converged"]
    print()
    
    # Check 4: Comparison distribution
    check_comparison_distribution(comparisons)
    print()
    
    # Check 5: High uncertainty samples
    results["uncertainty"] = check_high_uncertainty_samples(uncertainties, comparisons)
    print()
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print("="*60)
    print(f"Summary: {passed}/{total} checks passed")
    print("="*60)
    
    return results


"""
Integrated workflow for analyzing eval realism.

This script provides an end-to-end workflow that:
1. Takes an Inspect eval file or JSON file with samples
2. Places samples on the Grok 4 fast leaderboard
3. Analyzes evaluation-like features through clustering
4. Saves all results to a single timestamped output directory

Usage:
    python analyze_eval_realism.py <eval_file> [--output-name <name>]

Examples:
    # With .eval file
    python analyze_eval_realism.py logs/my_eval.eval
    
    # With JSON file  
    python analyze_eval_realism.py partial_transcripts/my_samples.json
    
    # With custom output name
    python analyze_eval_realism.py my_eval.eval --output-name my_analysis
    
    # With per-sample clustering enabled
    python analyze_eval_realism.py my_eval.eval --per-sample-clusters
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import os
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from sample_integration.py (in same directory)
from sample_integration import (
    load_calibration_ratings,
    integrate_new_sample,
    calculate_leaderboard_placement,
    save_aggregate_results,
    save_placement_summary
)

# Import functions from cluster_eval_features.py (in parent directory)
from cluster_eval_features import (
    extract_eval_reasons,
    get_openai_embeddings,
    create_topic_model,
    fit_and_reduce_topics,
    generate_report,
    generate_per_sample_clusters,
    EmbeddingAPIError,
    OpenRouterEmbeddings
)

# Import helper for extracting from .eval files (in parent directory)
from extract_prompts_from_inspect_log import extract_prompts_minimal


def create_output_directory(base_name: str = None) -> Path:
    """
    Create a timestamped output directory for this analysis run.
    
    Args:
        base_name: Optional base name for the directory
        
    Returns:
        Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if base_name:
        dir_name = f"{base_name}_{timestamp}"
    else:
        dir_name = f"eval_analysis_{timestamp}"
    
    # Save to sample_lab/eval_analysis_results
    script_dir = Path(__file__).parent
    output_dir = script_dir / "eval_analysis_results" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def load_samples_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load samples from either a .eval file or .json file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of sample dictionaries
    """
    if file_path.suffix.lower() == '.json':
        print(f"Loading samples from JSON file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"✓ Loaded {len(samples)} samples from JSON")
    else:
        print(f"Extracting samples from .eval file: {file_path}")
        samples = extract_prompts_minimal(str(file_path))
        print(f"✓ Extracted {len(samples)} samples from .eval log")
    
    # Normalize the message key: convert 'prompts' to 'input' if needed
    for sample in samples:
        if 'prompts' in sample and 'input' not in sample:
            sample['input'] = sample.pop('prompts')
    
    return samples


def run_leaderboard_placement(
    samples: List[Dict[str, Any]],
    leaderboard_path: str,
    api_key: str,
    model: str,
    output_dir: Path,
    output_name: str
) -> str:
    """
    Run leaderboard placement analysis on samples using parallel processing.
    
    Args:
        samples: List of sample dictionaries
        leaderboard_path: Path to the leaderboard JSON file
        api_key: OpenRouter API key
        model: Model to use for comparisons
        output_dir: Directory to save results
        output_name: Base name for output files
        
    Returns:
        Path to the saved aggregate results JSON file
    """
    print("\n" + "="*80)
    print("STEP 1: LEADERBOARD PLACEMENT ANALYSIS")
    print("="*80)
    
    # Load calibration ratings
    calibration_ratings = load_calibration_ratings(leaderboard_path)
    
    print(f"\nStarting parallel processing of {len(samples)} samples...")
    print("="*80 + "\n")
    
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
                  f"θ={theta_after:.2f} (Δ{theta_change:+.2f}), σ={sigma_after:.2f}", flush=True)
    
    # Define function to process a single sample
    def process_single_sample(sample_index_and_data):
        i, sample = sample_index_and_data
        sample_id = sample.get('id', f'sample_{i}')
        
        # Run adaptive integration with callback for individual comparison results
        results = integrate_new_sample(
            new_sample=sample,
            calibration_ratings=calibration_ratings,
            k_initial=32,
            max_comparisons=15,
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
    all_results = []
    failed_samples = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=len(samples)) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_sample, (i, sample)): sample 
            for i, sample in enumerate(samples, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed_count += 1
                
                with print_lock:
                    print(f"✓ Completed {completed_count}/{len(samples)}: {result['new_sample_id']} → "
                          f"Rank #{result['leaderboard_rank']}/{result['leaderboard_total']} "
                          f"({result['leaderboard_percentile']}th percentile)", flush=True)
                          
            except Exception as e:
                sample = futures[future]
                completed_count += 1
                sample_id = sample.get('id', 'unknown')
                failed_samples.append(sample_id)
                
                with print_lock:
                    print(f"\n{'!'*80}", flush=True)
                    print(f"✗ FAILED {completed_count}/{len(samples)}: {sample_id}", flush=True)
                    print(f"Error: {type(e).__name__}: {e}", flush=True)
                    print(f"{'!'*80}", flush=True)
                    print("Traceback:", flush=True)
                    traceback.print_exc()
                    print(f"{'!'*80}\n", flush=True)
    
    # Print summary
    print("\n" + "="*80)
    print("LEADERBOARD PLACEMENT SUMMARY")
    print("="*80)
    print(f"Total samples attempted: {len(samples)}")
    print(f"Successfully processed: {len(all_results)}")
    
    if failed_samples:
        print(f"Failed: {len(failed_samples)}")
        print(f"Failed sample IDs: {', '.join(failed_samples)}")
    
    if all_results:
        # Print placement statistics
        ranks = [r['leaderboard_rank'] for r in all_results]
        percentiles = [r['leaderboard_percentile'] for r in all_results]
        print(f"\nPlacement Distribution:")
        print(f"  Best: #{min(ranks)} ({max(percentiles)}th percentile)")
        print(f"  Worst: #{max(ranks)} ({min(percentiles)}th percentile)")
        print(f"  Median: #{sorted(ranks)[len(ranks)//2]} ({sorted(percentiles)[len(percentiles)//2]}th percentile)")
        
        # Additional statistics
        total_comparisons = sum(r['num_comparisons'] for r in all_results)
        avg_comparisons = total_comparisons / len(all_results)
        print(f"\nComparison Statistics:")
        print(f"  Total comparisons: {total_comparisons}")
        print(f"  Average comparisons per sample: {avg_comparisons:.1f}")
        
        # Convergence statistics
        converged = sum(1 for r in all_results if 'threshold' in r.get('convergence_reason', '').lower())
        print(f"\nConvergence Statistics:")
        print(f"  Reached uncertainty threshold: {converged}/{len(all_results)}")
        print(f"  Hit max comparisons: {len(all_results) - converged}/{len(all_results)}")
        
        # Save results to output directory
        print(f"\nSaving results to: {output_dir}")
        
        # Save aggregate results
        aggregate_path = save_aggregate_results(
            all_results,
            output_dir=str(output_dir),
            verbose=True,
            output_name=output_name
        )
        
        # Save placement summary
        save_placement_summary(
            all_results,
            output_dir=str(output_dir),
            output_name=output_name
        )
        
        return aggregate_path
    else:
        print("\n⚠ No samples were successfully processed!")
        return None


def run_clustering_analysis(
    aggregate_results_path: str,
    output_dir: Path,
    api_key: str,
    per_sample_clusters: bool = False
) -> None:
    """
    Run clustering analysis on eval awareness reasons.
    
    Args:
        aggregate_results_path: Path to the aggregate results JSON from leaderboard placement
        output_dir: Directory to save clustering results
        api_key: OpenRouter API key
        per_sample_clusters: If True, run clustering separately for each sample
    """
    print("\n" + "="*80)
    print("STEP 2: CLUSTERING ANALYSIS OF EVAL AWARENESS REASONS")
    print("="*80)
    
    # Extract evaluation reasons from JSON
    print(f"\nExtracting eval awareness reasons from: {aggregate_results_path}")
    docs, short_to_full, sample_ids = extract_eval_reasons(aggregate_results_path)
    
    if not docs:
        print("⚠ No eval awareness reasons found. Skipping clustering analysis.")
        print("This may happen if all samples were judged as realistic.")
        return
    
    total_arguments = len(docs)
    print(f"\nTotal arguments to cluster: {total_arguments}")
    print(f"Total unique samples: {len(set(sample_ids))}")
    
    # Hyperparameters for clustering
    n_neighbors = 15
    min_cluster_size = 5
    min_samples = 3
    min_df = 1
    random_state = 42
    
    print(f"\nClustering parameters:")
    print(f"  UMAP n_neighbors: {n_neighbors}")
    print(f"  HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"  HDBSCAN min_samples: {min_samples}")
    print(f"  CountVectorizer min_df: {min_df}")
    print(f"  Random state: {random_state}")
    
    # Get embeddings from OpenAI via OpenRouter
    print("\n" + "="*80)
    partial_info = None
    try:
        embeddings = get_openai_embeddings(docs, api_key)
    except EmbeddingAPIError as err:
        if err.partial_embeddings is None or err.processed_count == 0:
            print("\n✗ Embedding generation failed before any arguments were processed. Aborting clustering step.")
            return
        print(
            f"\nWARNING: Embedding generation failed after {err.processed_count}/{total_arguments} arguments.\n"
            "Proceeding with the successfully processed subset."
        )
        docs = docs[:err.processed_count]
        sample_ids = sample_ids[:err.processed_count]
        embeddings = err.partial_embeddings
        partial_info = {
            "error": str(err),
            "processed_arguments": err.processed_count,
            "total_arguments": total_arguments
        }
    
    output_filename = "eval_awareness_clusters_per_sample.json" if per_sample_clusters else "eval_awareness_clusters.json"
    output_path = output_dir / output_filename
    
    if per_sample_clusters:
        generate_per_sample_clusters(
            docs=docs,
            embeddings=embeddings,
            short_to_full=short_to_full,
            sample_ids=sample_ids,
            output_path=output_path,
            api_key=api_key,
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            min_df=min_df,
            random_state=random_state,
            partial_info=partial_info
        )
    else:
        embedding_model = OpenRouterEmbeddings(api_key)
        
        print("\nCreating topic model...")
        topic_model = create_topic_model(
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            min_df=min_df,
            random_state=random_state,
            embedding_model=embedding_model,
            api_key=api_key
        )
        
        topics, probs = fit_and_reduce_topics(topic_model, docs, embeddings)
        
        generate_report(
            topic_model,
            docs,
            topics,
            output_path,
            short_to_full,
            sample_ids,
            api_key,
            partial_info=partial_info
        )
    
    print("\n✓ Clustering analysis complete!", flush=True)


def main(eval_file: str, output_name: str = None, per_sample_clusters: bool = False):
    """
    Main workflow function.
    
    Args:
        eval_file: Path to the .eval or .json file
        output_name: Optional base name for output directory
        per_sample_clusters: If True, cluster evaluation arguments per sample
    """
    # Load environment variables from parent directory
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. "
                        "Create a .env file with your API key.")
    
    # Configuration
    model = "x-ai/grok-4-fast"  # Model for comparisons
    script_dir = Path(__file__).parent.parent
    leaderboard_path = script_dir / "leaderboards" / "grok_4_fast_cleaned_leaderboard_v1.json"
    
    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Grok 4 fast leaderboard not found at: {leaderboard_path}\n"
            f"Please ensure the leaderboard file exists or run calculate_leaderboard.py first."
        )
    
    # Resolve eval file path
    eval_file_path = Path(eval_file)
    if not eval_file_path.is_absolute() and not eval_file_path.exists():
        # Try looking in eval_logs/ (parent) or partial_transcripts/ (in sample_lab)
        potential_eval = script_dir / "eval_logs" / eval_file
        potential_json = Path(__file__).parent / "partial_transcripts" / eval_file
        
        if potential_eval.exists():
            eval_file_path = potential_eval
        elif potential_json.exists():
            eval_file_path = potential_json
        else:
            raise FileNotFoundError(
                f"Eval file not found at: {eval_file}\n"
                f"Also checked:\n"
                f"  - {potential_eval}\n"
                f"  - {potential_json}"
            )
    
    if not eval_file_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file_path}")
    
    # Create output directory
    if output_name is None:
        # Use eval file name as base
        output_name = eval_file_path.stem
    
    output_dir = create_output_directory(output_name)
    
    # Print workflow header
    print("="*80)
    print("EVAL REALISM ANALYSIS WORKFLOW")
    print("="*80)
    print(f"Input file: {eval_file_path}")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Step 1: Load samples
    samples = load_samples_from_file(eval_file_path)
    
    if not samples:
        print("\n⚠ No samples found in input file. Exiting.")
        return
    
    # Step 2: Run leaderboard placement
    aggregate_results_path = run_leaderboard_placement(
        samples=samples,
        leaderboard_path=str(leaderboard_path),
        api_key=api_key,
        model=model,
        output_dir=output_dir,
        output_name=output_name
    )
    
    if aggregate_results_path is None:
        print("\n⚠ Leaderboard placement failed. Skipping clustering analysis.")
        return
    
    # Step 3: Run clustering analysis
    run_clustering_analysis(
        aggregate_results_path=aggregate_results_path,
        output_dir=output_dir,
        api_key=api_key,
        per_sample_clusters=per_sample_clusters
    )
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. {output_name}.json - Full leaderboard placement data")
    print(f"  2. {output_name}_summary.txt - Human-readable placement summary")
    cluster_prefix = "eval_awareness_clusters_per_sample" if per_sample_clusters else "eval_awareness_clusters"
    mode_label = "Per-sample clustering data" if per_sample_clusters else "Global clustering data"
    print(f"  3. {cluster_prefix}.json - {mode_label}")
    print(f"  4. {cluster_prefix}_summary.txt - Clustering summary")
    print("="*80 + "\n")


def cli_main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze eval realism: leaderboard placement + clustering analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a .eval file
  python analyze_eval_realism.py logs/my_eval.eval
  
  # Analyze a JSON file with samples
  python analyze_eval_realism.py partial_transcripts/my_samples.json
  
  # Use custom output name
  python analyze_eval_realism.py my_eval.eval --output-name my_analysis
  
  # If file is in eval_logs/ or partial_transcripts/, just use filename
  python analyze_eval_realism.py my_eval.eval
  python analyze_eval_realism.py my_samples.json

Results will be saved to: eval_analysis_results/<output_name>_<timestamp>/
        """
    )
    
    parser.add_argument(
        'eval_file',
        type=str,
        help='Path to .eval log file with samples'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Base name for output directory (default: uses eval filename)'
    )
    
    parser.add_argument(
        '--per-sample-clusters',
        action='store_true',
        help='Cluster eval-awareness arguments per sample instead of across the dataset'
    )
    
    args = parser.parse_args()
    
    try:
        main(args.eval_file, args.output_name, per_sample_clusters=args.per_sample_clusters)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode
        cli_main()
    else:
        # Direct execution mode with example
        print("="*80)
        print("RUNNING IN DIRECT EXECUTION MODE (for testing)")
        print("="*80)
        print("To use CLI mode, run with arguments:")
        print("  python analyze_eval_realism.py <eval_file> [--output-name <name>]")
        print("="*80 + "\n")
        
        # Example usage - modify these as needed
        script_dir = Path(__file__).parent
        eval_file = script_dir / "eval_logs" / "oversight_subversion_basic_gpt_5_mini.eval"
        
        if eval_file.exists():
            main(str(eval_file), output_name="example_analysis")
        else:
            print(f"Example eval file not found: {eval_file}")
            print("Please provide an eval file path as command line argument.")
            sys.exit(1)


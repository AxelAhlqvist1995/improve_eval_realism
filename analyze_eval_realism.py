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
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import os

# Import functions from sample_integration.py
from sample_integration import (
    load_calibration_ratings,
    integrate_new_sample,
    calculate_leaderboard_placement,
    save_aggregate_results,
    save_placement_summary
)

# Import functions from cluster_eval_features.py
from cluster_eval_features import (
    extract_eval_reasons,
    get_openai_embeddings,
    create_topic_model,
    fit_and_reduce_topics,
    generate_report,
    OpenRouterEmbeddings
)

# Import helper for extracting from .eval files
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
    Run leaderboard placement analysis on samples.
    
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
    
    print(f"\nProcessing {len(samples)} samples...")
    print("="*80 + "\n")
    
    # Process all samples
    all_results = []
    failed_samples = []
    
    for i, sample in enumerate(samples, 1):
        sample_id = sample.get('id', f'sample_{i}')
        
        try:
            print(f"[{i}/{len(samples)}] Processing: {sample_id}")
            
            # Run adaptive integration
            results = integrate_new_sample(
                new_sample=sample,
                calibration_ratings=calibration_ratings,
                k_initial=32,
                max_comparisons=1,
                uncertainty_threshold=20.0,
                api_key=api_key,
                model=model,
                k_neighbors=10,
                verbose=True
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
            
            all_results.append(results)
            
            print(f"✓ Placement: Rank #{rank}/{total} ({percentile}th percentile)\n")
            
        except Exception as e:
            print(f"✗ Failed to process {sample_id}: {e}\n")
            failed_samples.append(sample_id)
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("LEADERBOARD PLACEMENT SUMMARY")
    print("="*80)
    print(f"Successfully processed: {len(all_results)}/{len(samples)}")
    
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
    api_key: str
) -> None:
    """
    Run clustering analysis on eval awareness reasons.
    
    Args:
        aggregate_results_path: Path to the aggregate results JSON from leaderboard placement
        output_dir: Directory to save clustering results
        api_key: OpenRouter API key
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
    
    print(f"\nTotal arguments to cluster: {len(docs)}")
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
    embeddings = get_openai_embeddings(docs, api_key)
    
    # Create embedding model for BERTopic
    embedding_model = OpenRouterEmbeddings(api_key)
    
    # Create topic model
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
    
    # Fit and reduce topics
    topics, probs = fit_and_reduce_topics(topic_model, docs, embeddings)
    
    # Generate report with LLM-based topic labels
    output_path = output_dir / "eval_awareness_clusters.json"
    report_data = generate_report(
        topic_model,
        docs,
        topics,
        output_path,
        short_to_full,
        sample_ids,
        api_key
    )
    
    print("\n✓ Clustering analysis complete!")


def main(eval_file: str, output_name: str = None):
    """
    Main workflow function.
    
    Args:
        eval_file: Path to the .eval or .json file
        output_name: Optional base name for output directory
    """
    # Load environment variables
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. "
                        "Create a .env file with your API key.")
    
    # Configuration
    model = "x-ai/grok-4-fast"  # Model for comparisons
    script_dir = Path(__file__).parent
    leaderboard_path = script_dir / "leaderboards" / "grok_4_fast_cleaned_leaderboard_v1.json"
    
    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Grok 4 fast leaderboard not found at: {leaderboard_path}\n"
            f"Please ensure the leaderboard file exists or run calculate_leaderboard.py first."
        )
    
    # Resolve eval file path
    eval_file_path = Path(eval_file)
    if not eval_file_path.is_absolute() and not eval_file_path.exists():
        # Try looking in eval_logs/ or partial_transcripts/
        potential_eval = script_dir / "eval_logs" / eval_file
        potential_json = script_dir / "partial_transcripts" / eval_file
        
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
        api_key=api_key
    )
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. {output_name}.json - Full leaderboard placement data")
    print(f"  2. {output_name}_summary.txt - Human-readable placement summary")
    print(f"  3. eval_awareness_clusters.json - Clustering analysis data")
    print(f"  4. eval_awareness_clusters_summary.txt - Clustering summary")
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
    
    args = parser.parse_args()
    
    try:
        main(args.eval_file, args.output_name)
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
        eval_file = script_dir / "logs" / "2025-10-19T14-32-43+03-00_oversight-subversion_2s8d5NcbFfm9ffNrzJZyjt.eval"
        
        if eval_file.exists():
            main(str(eval_file), output_name="example_analysis")
        else:
            print(f"Example eval file not found: {eval_file}")
            print("Please provide an eval file path as command line argument.")
            sys.exit(1)


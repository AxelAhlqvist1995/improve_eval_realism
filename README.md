# AI Conversation Realism Evaluator

A system for evaluating and ranking AI conversation transcripts by their realism using the Bradley-Terry rating model with LLM-based pairwise comparisons.

## Overview

This project implements a comprehensive evaluation framework to determine how realistic AI assistant conversations appear. It uses multiple LLM judges to perform pairwise comparisons between conversation transcripts, then applies the Bradley-Terry model to compute global ratings and rankings.

## Setup

### Setup

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

4. Unizp datasets/processed_needham_dataset.json.zip
It's zipped in order to respect licensing requirements of some samples used for this dataset


## Quick Start

> **Note:** Sample-lab utilities live inside `sample_lab/`. Run them from the
> repository root via `python sample_lab/<script>.py`. The commands below use
> that convention.

### 0. Unzip the dataset

Unzip processe_cleaned_needham_dataset.json.zip (it's zipped for licensing reasons)

### 1. Calculate a Leaderboard from Scratch

```bash
python calculate_leaderboard.py
```

This will:
- Load samples from `datasets/processed_cleaned_needham_dataset.json`
- Run adaptive pairwise comparisons (exploration then exploitation phases)
- Calculate Bradley-Terry ratings
- Save results to `leaderboards/bradley_terry_results_<timestamp>.json`

Configure parameters in the `main()` function:
```python
results = run_bradley_terry_competition(
    samples=agentic_samples,
    api_key=api_key,
    model="x-ai/grok-4-fast",
    max_rounds=16,              # Number of comparison rounds
    max_uncertainty_threshold=20.0,  # Stop when uncertainty is low enough
    max_workers=600             # Parallel workers
)
```

### 2. Integrate New Samples into Existing Leaderboard

```bash
python sample_lab/sample_integration.py <input_file> --model <model_name>
```

The script accepts both `.eval` files and `.json` files as input:

**From an inspect eval log:**
```bash
python sample_lab/sample_integration.py 2025-10-19T14-32-43+03-00_oversight-subversion.eval \
    --model openai/gpt-5-mini \
    --leaderboard leaderboards/gpt_5_mini_leaderboard.json \
    --output my_results
```

**From a JSON file with samples:**
```bash
python sample_lab/sample_integration.py sample_lab/partial_transcripts/partial_transcripts_swe_gym_4458.json \
    --model x-ai/grok-4-fast \
    --leaderboard leaderboards/grok_4_fast_leaderboard.json \
    --output swe_gym_4458
```

**Options:**
- `--model` / `-m`: OpenRouter model for comparisons (required)
- `--leaderboard`: Path to leaderboard JSON file (defaults based on model)
- `--output` / `-o`: Base name for output files (auto-generated if omitted)

**What it does:**
1. Extracts/loads samples from the input file
2. For each sample, runs comparisons against a leaderboard
3. Calculates leaderboard placement (rank and percentile)
4. Saves detailed results to `sample_lab/samples_mapped_to_leaderboard/`:
   - `<output_name>.json`: Full comparison data with judge reasoning
   - `<output_name>_summary.txt`: Human-readable placement summary

### 3. Analyze Eval Realism (Integrated Workflow)

**NEW: Complete end-to-end workflow** that combines leaderboard placement and clustering analysis:

```bash
python sample_lab/analyze_eval_realism.py <eval_file> [--output-name <name>]
```

**Examples:**
```bash
# Analyze a .eval file
python sample_lab/analyze_eval_realism.py sample_lab/eval_logs/my_eval.eval

# Analyze a JSON file with samples
python sample_lab/analyze_eval_realism.py sample_lab/partial_transcripts/my_samples.json

# Use custom output name
python sample_lab/analyze_eval_realism.py sample_lab/eval_logs/my_eval.eval --output-name my_analysis
```

**What it does:**
1. Loads samples from `.eval` file
2. Places each sample on the Grok 4 fast leaderboard
3. Extracts eval awareness reasons from judge comparisons
4. Clusters similar reasons using BERTopic with LLM-based topic labels
5. Saves all results to a single timestamped directory

**Outputs:**
All files are saved to `sample_lab/eval_analysis_results/<output_name>_<timestamp>/`:
- `<output_name>.json`: Full leaderboard placement data with judge reasoning
- `<output_name>_summary.txt`: Human-readable placement summary
- `eval_awareness_clusters.json`: Complete cluster data
- `eval_awareness_clusters_summary.txt`: Cluster summary with top patterns

This is the **recommended workflow** for analyzing eval realism of an Inspect eval log in one command!


---

### 3a. Cluster Evaluation Features (Standalone)

For standalone clustering analysis of existing placement results:

```bash
python sample_lab/cluster_eval_features.py
```

By default, this processes `sample_lab/samples_mapped_to_leaderboard/explanations_test.json` and saves results to `sample_lab/clusters/explanations_test_clusters.json`. You can modify the input file in the script's `main()` function.

**What it does:**
1. Extracts judge arguments from comparison data where samples appeared evaluation-like
2. Uses BERTopic with embeddings to cluster similar arguments
3. Generates human-readable topic labels using LLM
4. Identifies common patterns in evaluation-like features

**Outputs:**
- `<input_name>_clusters.json`: Full cluster data with all arguments
  - Each topic includes name, count, and all arguments (short + full descriptions)
  - Arguments include sample_id for traceability
- `<input_name>_clusters_summary.txt`: Human-readable summary
  - Top 10 topics with representative examples
  - Useful for quick overview of evaluation patterns


### 4. Compare Evaluation Files (Dumbbell Plots)

Visualize how the same samples perform across different evaluation configurations:

```bash
python sample_lab/compare_leaderboard_placements.py <file1.json> <file2.json> [file3.json ...] [options]
```

**Examples:**

```bash
# Compare two evaluations (files automatically looked up in samples_mapped_to_leaderboard/)
python sample_lab/compare_leaderboard_placements.py eval_a.json eval_b.json
  
# Add custom title
python sample_lab/compare_leaderboard_placements.py file1.json file2.json -t "Impact of Configuration Change"

# You can also use full paths if files are elsewhere
python sample_lab/compare_leaderboard_placements.py /path/to/file1.json /path/to/file2.json
```

**What it does:**
1. Loads multiple JSON files (automatically looks in `samples_mapped_to_leaderboard/` if only filenames provided)
2. Identifies samples with identical IDs across all files
3. Creates a dumbbell plot showing each sample's percentile rank across evaluations
4. Color-codes each evaluation for easy comparison
5. Saves plot to `sample_lab/plots/eval_file_comparison/`

**Options:**
- `-o, --output`: Output filename (without extension). Defaults to timestamp.
- `-t, --title`: Custom title for the plot

**Output:**
- `<output_name>.png`: Dumbbell plot visualization (300 DPI)

**Use cases:**
- Compare impact of system prompts or model changes
- Track sample performance across different leaderboards
- Identify which samples benefit most from specific configurations
- Validate consistency of evaluation approaches

### 5. Run Sanity Checks

The project includes several sanity checks to validate leaderboards and rating systems.

**Compare Two Leaderboards:**

```bash
python sanity_checks/leaderboard_correlation_test.py <leaderboard1.json> <leaderboard2.json>
```

Example:
```bash
python sanity_checks/leaderboard_correlation_test.py \
    leaderboards/grok_4_fast_leaderboard.json \
    leaderboards/grok_4_fast_cleaned_leaderboard_v1.json
```

**What it does:**
1. Computes Spearman rank correlation between two leaderboards
2. Identifies common samples and compares their rankings
3. Outputs correlation coefficient, p-value, and significance
4. Displays an interactive scatter plot showing ranking correlation

**Use cases:**
- Validate consistency between leaderboards from different models
- Check if cleaning/processing affected rankings
- Compare leaderboards computed at different times
- Assess impact of different comparison strategies


## Project Structure

```
Improve eval realism/
├── README.md                          # This file
├── .env                               # API keys (not in repo)
│
├── Core Rating System
│   ├── bradley_terry.py               # Bradley-Terry MM algorithm implementation
│   ├── calculate_leaderboard.py      # Full leaderboard calculation pipeline
│   └── sample_comparison.py          # LLM-based pairwise comparison logic
│
├── Sample Lab (analysis workflows)
│   ├── analyze_eval_realism.py       # ⭐ Integrated workflow (placement + clustering)
│   ├── sample_integration.py         # Adaptive integration for new samples
│   ├── cluster_eval_features.py      # BERTopic clustering of evaluation features
│   ├── compare_leaderboard_placements.py  # Compare samples across multiple evaluations
│   ├── plot_turns_vs_percentile.py   # Visualization utilities
│   ├── create_partial_transcripts.py # Create partial transcript samples
│   ├── aggregate_existing_results.py # Combine integration results
│   ├── partial_transcripts/          # Processed transcript samples
│   ├── samples_mapped_to_leaderboard/   # Integration results with judge reasoning
│   ├── eval_analysis_results/        # Results from analyze_eval_realism
│   ├── clusters/                     # Topic clustering results
│   └── plots/                        # Visualization outputs
│       ├── eval_file_comparison/     # Dumbbell plots comparing evaluations
│       └── ...
│
├── Data Processing
│   ├── process_needham_dataset.py    # Dataset preprocessing utilities
│   └── extract_prompts_from_inspect_log.py  # Extract samples from eval logs
│
├── Quality Assurance
│   └── sanity_checks/
│       ├── basic_sanity_checks.py        # Automated validation checks
│       ├── diagnose_bradley_terry_convergence.py  # Convergence diagnostics
│       ├── transitivity_test.py          # Test logical consistency
│       ├── order_consistency_test.py     # Test for positional bias
│       └── leaderboard_correlation_test.py # Compare leaderboard rankings
│
├── Utilities
│   └── SAMPLE_INTEGRATION_GUIDE.md   # Detailed integration guide
│
└── Data & Results
    ├── datasets/                      # Input conversation datasets
    ├── leaderboards/                  # Calculated leaderboards
    └── eval_logs/                     # Root-level eval log files
```

## How It Works

### 1. Pairwise Comparison with Multiple Judges

Each comparison between two transcripts uses **5 different judge prompts**, each applied in **both orders** (transcript A vs B, then B vs A), resulting in 10 independent judgments:

This design:
- Reduces bias from any single prompt
- Mitigates order effects
- Provides more robust decisions

### 2. Bradley-Terry Rating Model

The Bradley-Terry model estimates each sample's "strength" (θ) using the probability:

```
P(sample_i beats sample_j) = θ_i / (θ_i + θ_j)
```

We use the **Minorization-Maximization (MM) algorithm** to efficiently compute ratings from comparison results:


Ratings are on a log scale, centered at 0, with typical ranges of -500 to +500.

### 3. Uncertainty Estimation

Each rating has an associated uncertainty value (σ) computed using **Bayesian posterior variance updates**:
Uncertainty decreases as more comparisons are made, especially for close matchups.

### 4. Adaptive Integration Strategy

When adding new samples to an existing leaderboard:

1. **Initialize** at the median rating of the calibration set
2. **Find k-nearest neighbors** based on current rating estimate
3. **Select opponent** prioritizing high-confidence (low σ) calibration samples
4. **Run comparison** and update rating using Bradley-Terry
5. **Repeat** until uncertainty drops below threshold or max comparisons reached

This typically requires only 10-18 comparisons per sample vs. comparing against all calibration samples.


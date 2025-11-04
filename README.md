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
python sample_integration.py <input_file> --model <model_name>
```

The script accepts both `.eval` files and `.json` files as input:

**From an inspect eval log:**
```bash
python sample_integration.py 2025-10-19T14-32-43+03-00_oversight-subversion.eval \
    --model openai/gpt-5-mini \
    --leaderboard leaderboards/gpt_5_mini_leaderboard.json \
    --output my_results
```

**From a JSON file with samples:**
```bash
python sample_integration.py partial_transcripts/partial_transcripts_swe_gym_4458.json \
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
4. Saves detailed results to `apollo_leaderboard_placement/`:
   - `<output_name>.json`: Full comparison data with judge reasoning
   - `<output_name>_summary.txt`: Human-readable placement summary

### 3. Cluster Evaluation Features

Analyze what makes conversations appear evaluation-like using topic modeling:

```bash
python cluster_eval_features.py
```

By default, this processes `apollo_leaderboard_placement/explanations_test.json` and saves results to `clusters/explanations_test_clusters.json`. You can modify the input file in the script's `main()` function.

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



## Project Structure

```
Improve eval realism/
├── README.md                          # This file
├── .env                               # API keys (not in repo)
│
├── Core Rating System
│   ├── bradley_terry.py               # Bradley-Terry MM algorithm implementation
│   ├── calculate_leaderboard.py      # Full leaderboard calculation pipeline
│   ├── sample_comparison.py          # LLM-based pairwise comparison logic
│   └── sample_integration.py         # Adaptive integration for new samples
│
├── Analysis Tools
│   ├── cluster_eval_features.py      # BERTopic clustering of evaluation features
│   ├── plot_turns_vs_percentile.py   # Visualization utilities
│
├── Data Processing
│   ├── process_needham_dataset.py    # Dataset preprocessing utilities
│   ├── extract_prompts_from_inspect_log.py  # Extract samples from eval logs
│   └── create_partial_transcripts.py # Create partial transcript samples
│
├── Quality Assurance
│   └── sanity_checks/
│       ├── basic_sanity_checks.py        # Automated validation checks
│       ├── diagnose_bradley_terry_convergence.py  # Convergence diagnostics
│       ├── transitivity_test.py      # Test logical consistency
│       └── order_consistency_test.py # Test for positional bias
│
├── Utilities
│   ├── aggregate_existing_results.py # Combine integration results
│   └── SAMPLE_INTEGRATION_GUIDE.md   # Detailed integration guide
│
└── Data & Results
    ├── datasets/                      # Input conversation datasets
    ├── leaderboards/                  # Calculated leaderboards
    ├── eval_logs/                     # Raw eval log files
    ├── apollo_leaderboard_placement/  # Integration results with judge reasoning
    ├── clusters/                      # Topic clustering results
    ├── partial_transcripts/           # Processed transcript samples
    └── plots/                         # Visualization outputs
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


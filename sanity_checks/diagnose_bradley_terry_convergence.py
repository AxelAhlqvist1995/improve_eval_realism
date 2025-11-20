"""
Diagnostic script to analyze why Bradley-Terry MM algorithm failed to converge.
Optimized for speed and low memory usage.
"""

import json
import math
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime


class DiagnosticWriter:
    """Write output to both console and file without storing in memory."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
    
    def log(self, msg):
        print(msg, flush=True)
        self.file.write(msg + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()


def load_results(results_path: str) -> Tuple[Dict, List[Dict]]:
    """Load only essential data from JSON file."""
    print("  Loading JSON file...", flush=True)
    with open(results_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    final_ratings = full_data['final_ratings']
    comparison_history = full_data['comparison_history']
    
    # Clear large data immediately
    del full_data
    
    print(f"  ✓ Loaded {len(final_ratings)} ratings, {len(comparison_history)} comparisons", flush=True)
    return final_ratings, comparison_history


def check_disconnected_graph(comparisons: List[Dict], writer: DiagnosticWriter) -> bool:
    """Check if comparison graph is disconnected."""
    writer.log("\n" + "="*80)
    writer.log("1. DISCONNECTED COMPARISON GRAPH CHECK")
    writer.log("="*80)
    
    # Build graph
    G = nx.Graph()
    for comp in comparisons:
        G.add_edge(comp['sample_a'], comp['sample_b'])
    
    writer.log(f"\nTotal samples (nodes): {G.number_of_nodes()}")
    writer.log(f"Total comparisons (edges): {G.number_of_edges()}")
    
    components = list(nx.connected_components(G))
    writer.log(f"\nNumber of connected components: {len(components)}")
    
    if len(components) == 1:
        writer.log("✓ Graph is fully connected - NOT the cause")
        return False
    else:
        writer.log(f"✗ PROBLEM: Graph has {len(components)} disconnected components!")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5], 1):
            writer.log(f"  Component {i}: {len(comp)} samples")
        return True


def check_perfect_separation(comparisons: List[Dict], writer: DiagnosticWriter) -> Tuple[bool, List]:
    """Check for perfect separation - samples with 100% or 0% win rates."""
    writer.log("\n" + "="*80)
    writer.log("2. PERFECT SEPARATION / EXTREME WIN RATES CHECK")
    writer.log("="*80)
    
    # Calculate win/loss stats
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})
    for comp in comparisons:
        a, b, result = comp['sample_a'], comp['sample_b'], comp['result']
        if result == 'A':
            stats[a]['wins'] += 1
            stats[b]['losses'] += 1
        elif result == 'B':
            stats[a]['losses'] += 1
            stats[b]['wins'] += 1
        else:
            stats[a]['ties'] += 1
            stats[b]['ties'] += 1
    
    perfect_100, perfect_0 = [], []
    near_high, near_low = [], []
    
    for sid, s in stats.items():
        total = s['wins'] + s['losses']
        if total > 0:
            win_rate = s['wins'] / total
            if win_rate == 1.0:
                perfect_100.append((sid, s))
            elif win_rate == 0.0:
                perfect_0.append((sid, s))
            elif win_rate > 0.95:
                near_high.append((sid, s, win_rate))
            elif win_rate < 0.05:
                near_low.append((sid, s, win_rate))
    
    writer.log(f"\nSamples with 100% win rate: {len(perfect_100)}")
    if perfect_100:
        writer.log("✗ PROBLEM FOUND: Perfect winners detected!")
        for sid, s in perfect_100:
            writer.log(f"  {sid}: {s['wins']}W-{s['losses']}L-{s['ties']}T")
    
    writer.log(f"\nSamples with 0% win rate: {len(perfect_0)}")
    if perfect_0:
        writer.log("✗ PROBLEM FOUND: Perfect losers detected!")
        for sid, s in perfect_0:
            writer.log(f"  {sid}: {s['wins']}W-{s['losses']}L-{s['ties']}T")
    
    writer.log(f"\nSamples with >95% win rate: {len(near_high)}")
    if near_high and len(near_high) <= 15:
        for sid, s, wr in sorted(near_high, key=lambda x: x[2], reverse=True):
            writer.log(f"  {sid}: {s['wins']}W-{s['losses']}L-{s['ties']}T ({wr:.1%})")
    
    writer.log(f"\nSamples with <5% win rate: {len(near_low)}")
    if near_low and len(near_low) <= 15:
        for sid, s, wr in sorted(near_low, key=lambda x: x[2]):
            writer.log(f"  {sid}: {s['wins']}W-{s['losses']}L-{s['ties']}T ({wr:.1%})")
    
    # Quick cycle check (limited to avoid memory issues)
    writer.log("\nChecking for circular dominance patterns (sample)...")
    G_dir = nx.DiGraph()
    for comp in comparisons:
        if comp['result'] == 'A':
            G_dir.add_edge(comp['sample_a'], comp['sample_b'])
        elif comp['result'] == 'B':
            G_dir.add_edge(comp['sample_b'], comp['sample_a'])
    
    # Find just 3-cycles (triangles) - fast and memory-efficient
    cycles_found = 0
    nodes = list(G_dir.nodes())[:50]  # Check only first 50 nodes
    for node in nodes:
        for n1 in G_dir.successors(node):
            for n2 in G_dir.successors(n1):
                if G_dir.has_edge(n2, node):
                    cycles_found += 1
                    if cycles_found == 1:
                        writer.log(f"  Example cycle: {node} -> {n1} -> {n2} -> {node}")
                    if cycles_found >= 5:
                        break
            if cycles_found >= 5:
                break
        if cycles_found >= 5:
            break
    
    if cycles_found > 0:
        writer.log(f"✓ Found circular dominance (at least {cycles_found} 3-cycles)")
    else:
        writer.log("✗ PROBLEM: No circular dominance found")
    
    problem_samples = [s[0] for s in perfect_100 + perfect_0]
    has_problems = len(perfect_100) > 0 or len(perfect_0) > 0
    return has_problems, problem_samples


def check_sparse_comparisons(comparisons: List[Dict], writer: DiagnosticWriter) -> bool:
    """Check for sparse comparisons."""
    writer.log("\n" + "="*80)
    writer.log("3. SPARSE COMPARISONS CHECK")
    writer.log("="*80)
    
    comp_counts = defaultdict(int)
    for comp in comparisons:
        comp_counts[comp['sample_a']] += 1
        comp_counts[comp['sample_b']] += 1
    
    counts = list(comp_counts.values())
    
    writer.log(f"\nComparison count statistics:")
    writer.log(f"  Min: {min(counts)}, Max: {max(counts)}")
    writer.log(f"  Mean: {np.mean(counts):.2f}, Median: {np.median(counts):.2f}")
    writer.log(f"  Std Dev: {np.std(counts):.2f}")
    
    if len(set(counts)) == 1:
        writer.log(f"\n✓ All samples have exactly {counts[0]} comparisons (balanced)")
    else:
        writer.log(f"\n⚠ Unbalanced: varies from {min(counts)} to {max(counts)}")
    
    return len([c for c in counts if c < 3]) > 0


def check_tie_frequency(comparisons: List[Dict], writer: DiagnosticWriter) -> bool:
    """Check tie frequency."""
    writer.log("\n" + "="*80)
    writer.log("4. TIE FREQUENCY CHECK")
    writer.log("="*80)
    
    result_counts = Counter(comp['result'] for comp in comparisons)
    total = len(comparisons)
    tie_rate = result_counts.get('tie', 0) / total
    
    writer.log(f"\nOverall comparison results:")
    writer.log(f"  A wins: {result_counts.get('A', 0)} ({result_counts.get('A', 0)/total*100:.1f}%)")
    writer.log(f"  B wins: {result_counts.get('B', 0)} ({result_counts.get('B', 0)/total*100:.1f}%)")
    writer.log(f"  Ties: {result_counts.get('tie', 0)} ({tie_rate*100:.1f}%)")
    
    if tie_rate > 0.3:
        writer.log(f"\n✗ PROBLEM: Tie rate of {tie_rate*100:.1f}% is high (>30%)")
        return True
    else:
        writer.log(f"\n✓ Tie rate is acceptable")
        return False


def check_data_quality(comparisons: List[Dict], final_ratings: Dict, writer: DiagnosticWriter) -> bool:
    """Check data quality."""
    writer.log("\n" + "="*80)
    writer.log("5. DATA QUALITY CHECKS")
    writer.log("="*80)
    
    writer.log(f"\nBasic counts:")
    writer.log(f"  Total samples: {len(final_ratings)}")
    writer.log(f"  Total comparisons: {len(comparisons)}")
    
    # Self-comparisons
    self_comps = sum(1 for c in comparisons if c['sample_a'] == c['sample_b'])
    writer.log(f"  Self-comparisons: {self_comps}")
    if self_comps > 0:
        writer.log("  ✗ PROBLEM: Found self-comparisons!")
    else:
        writer.log("  ✓ No self-comparisons")
    
    # Duplicates
    comp_pairs = defaultdict(int)
    for comp in comparisons:
        pair = tuple(sorted([comp['sample_a'], comp['sample_b']]))
        comp_pairs[pair] += 1
    
    duplicates = sum(1 for v in comp_pairs.values() if v > 1)
    writer.log(f"  Duplicate comparisons: {duplicates}")
    
    # Density
    min_comps = len(final_ratings) - 1
    writer.log(f"\nComparison density:")
    writer.log(f"  Minimum for connectivity: {min_comps}")
    writer.log(f"  Actual: {len(comparisons)}")
    writer.log(f"  Ratio: {len(comparisons) / min_comps:.2f}x minimum")
    
    return self_comps > 0


def simulate_convergence(comparisons: List[Dict], writer: DiagnosticWriter) -> bool:
    """Simulate MM convergence."""
    writer.log("\n" + "="*80)
    writer.log("6. CONVERGENCE PATTERN SIMULATION")
    writer.log("="*80)
    
    # Get unique samples
    all_samples = list(set(c['sample_a'] for c in comparisons) | 
                       set(c['sample_b'] for c in comparisons))
    
    writer.log(f"\nSimulating MM algorithm with {len(all_samples)} samples...")
    writer.log("Using regularization lambda=0.5")
    
    ratings = {sid: 1.0 for sid in all_samples}
    lambda_reg = 0.5
    
    for iteration in range(50):
        old_ratings = ratings.copy()
        
        # MM update
        for sample_i in all_samples:
            wins = sum(1 for c in comparisons 
                      if (c['sample_a'] == sample_i and c['result'] == 'A') or 
                         (c['sample_b'] == sample_i and c['result'] == 'B'))
            ties = sum(1 for c in comparisons 
                      if (c['sample_a'] == sample_i or c['sample_b'] == sample_i) and 
                         c['result'] == 'tie')
            
            denom = lambda_reg
            for c in comparisons:
                if c['sample_a'] == sample_i:
                    denom += 1.0 / (old_ratings[sample_i] + old_ratings[c['sample_b']])
                elif c['sample_b'] == sample_i:
                    denom += 1.0 / (old_ratings[sample_i] + old_ratings[c['sample_a']])
            
            ratings[sample_i] = (wins + 0.5 * ties + lambda_reg) / denom
        
        # Normalize
        mean_rating = sum(ratings.values()) / len(ratings)
        old_mean = sum(old_ratings.values()) / len(old_ratings)
        old_ratings = {k: v / old_mean for k, v in old_ratings.items()}
        ratings = {k: v / mean_rating for k, v in ratings.items()}
        
        max_change = max(abs(ratings[k] - old_ratings[k]) for k in ratings)
        
        if iteration < 10 or iteration % 10 == 9:
            writer.log(f"  Iteration {iteration+1:3d}: max_change = {max_change:.6f}")
    
    writer.log("\nConvergence analysis:")
    if max_change < 0.01:
        writer.log("  ✓ CONVERGED: Algorithm reached tolerance threshold")
        return True
    else:
        writer.log(f"  ✗ NOT CONVERGING: max_change = {max_change:.6f} after 50 iterations")
        return False


def main():
    """Run all diagnostics."""
    # Get project root directory (parent of sanity_checks/)
    script_dir = Path(__file__).parent.parent
    results_file = script_dir / "leaderboards" / "grok_4_fast.json"
    report_file = script_dir / "mm_convergence_diagnostics" / f"convergence_diagnostic_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    
    # Create writer
    writer = DiagnosticWriter(report_file)
    
    writer.log("="*80)
    writer.log("BRADLEY-TERRY MM CONVERGENCE DIAGNOSTIC REPORT")
    writer.log("="*80)
    writer.log(f"Analyzing: {results_file.name}")
    writer.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    writer.log("\nLoading results...")
    final_ratings, comparisons = load_results(results_file)
    
    # Run diagnostics
    problems_found = []
    problem_samples = []
    
    # 1. Disconnected graph
    if check_disconnected_graph(comparisons, writer):
        problems_found.append("Disconnected comparison graph")
    
    # 2. Perfect separation
    has_sep, sep_samples = check_perfect_separation(comparisons, writer)
    if has_sep:
        problems_found.append("Perfect separation / extreme win rates")
        problem_samples = sep_samples
    
    # 3. Sparse comparisons
    if check_sparse_comparisons(comparisons, writer):
        problems_found.append("Very sparse comparisons")
    
    # 4. Tie frequency
    if check_tie_frequency(comparisons, writer):
        problems_found.append("High tie frequency")
    
    # 5. Data quality
    if check_data_quality(comparisons, final_ratings, writer):
        problems_found.append("Data quality issues")
    
    # 6. Convergence simulation
    if not simulate_convergence(comparisons, writer):
        problems_found.append("Failed to converge in simulation")
    
    # Summary
    writer.log("\n" + "="*80)
    writer.log("SUMMARY")
    writer.log("="*80)
    
    if problems_found:
        writer.log("\n✗ PROBLEMS FOUND:")
        for i, problem in enumerate(problems_found, 1):
            writer.log(f"  {i}. {problem}")
        
        writer.log("\n★ PRIMARY DIAGNOSIS:")
        if "Perfect separation" in problems_found[0] if problems_found else False:
            writer.log("  → PERFECT SEPARATION is the ROOT CAUSE of convergence failure")
            writer.log("  → The MM algorithm cannot converge when samples have 100% or 0% win rates")
            writer.log("  → These samples' ratings diverge to +∞ or -∞")
        
        writer.log("\nRECOMMENDED FIXES:")
        if problem_samples:
            writer.log("  1. Add REGULARIZATION: Use lambda > 0 (e.g., 0.5 or 1.0)")
            writer.log("     This prevents ratings from diverging to infinity")
            writer.log("  2. Remove outliers: Exclude perfect win/loss samples:")
            writer.log(f"     Samples to remove: {problem_samples}")
            writer.log("  3. Add virtual comparisons: Add weak ties for these samples")
    else:
        writer.log("\n✓ No major problems detected!")
        writer.log("  Consider: increasing max_iterations or adding regularization")
    
    writer.log("\n" + "="*80)
    writer.log(f"Report saved to: {report_file}")
    writer.log("="*80)
    
    writer.close()


if __name__ == "__main__":
    main()

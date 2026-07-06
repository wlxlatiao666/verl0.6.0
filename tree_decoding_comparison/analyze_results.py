#!/usr/bin/env python3
"""
Analyze and visualize results from Tree Decoding vs Base GRPO comparison.

Usage:
    python analyze_results.py --results-dir ./tree_decoding_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load results from directory."""
    summary_file = results_dir / "results_summary.json"
    generations_file = results_dir / "generations.json"

    results = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)

    if generations_file.exists():
        with open(generations_file, 'r') as f:
            results['generations'] = json.load(f)

    return results


def print_detailed_summary(results: Dict[str, Any]):
    """Print detailed summary of results."""
    if 'summary' not in results:
        print("No summary found")
        return

    summary = results['summary']
    base = summary['base_grpo']
    tree = summary['tree_decoding']
    config = summary['config']

    print("=" * 80)
    print("TREE DECODING vs BASE GRPO - DETAILED ANALYSIS")
    print("=" * 80)

    print("\nCONFIGURATION")
    print("-" * 40)
    print(f"  Model:               {config.get('model', 'N/A')}")
    print(f"  Number of samples:   {config.get('num_samples', 'N/A')}")
    print(f"  Sequences per query: {config.get('n', 'N/A')}")
    print(f"  Branching factor:    {config.get('branching_factor', 'N/A')}")
    print(f"  Max tree depth:      {config.get('max_tree_depth', 'N/A')}")
    print(f"  Entropy threshold:   {config.get('entropy_threshold', 'N/A')}")
    print(f"  Temperature:         {config.get('temperature', 'N/A')}")
    print(f"  Max tokens:          {config.get('max_tokens', 'N/A')}")

    print("\nPASS@K COMPARISON")
    print("-" * 40)

    print(f"\n{'k':<4} {'Base GRPO':<12} {'Tree Decoding':<12} {'Abs Improv':<12} {'Rel Improv':<12}")
    print("-" * 52)

    n = config.get('n', 8)
    base_pass_k = base.get('pass_k', {})
    tree_pass_k = tree.get('pass_k', {})

    best_k = None
    best_improvement = -1

    for k in range(1, n + 1):
        base_score = base_pass_k.get(f'pass@{k}', 0)
        tree_score = tree_pass_k.get(f'pass@{k}', 0)
        abs_imp = tree_score - base_score
        rel_imp = (abs_imp / base_score * 100) if base_score > 0 else 0

        if abs_imp > best_improvement:
            best_improvement = abs_imp
            best_k = k

        print(f"{k:<4} {base_score:<12.4f} {tree_score:<12.4f} {abs_imp:<+12.4f} {rel_imp:<+12.1f}%")

    print("\nTIMING")
    print("-" * 40)
    base_time = base.get('time', 0)
    tree_time = tree.get('time', 0)
    print(f"  Base GRPO:     {base_time:.2f}s")
    print(f"  Tree Decoding: {tree_time:.2f}s")
    print(f"  Difference:    {tree_time - base_time:+.2f}s")
    print(f"  Ratio:         {tree_time / base_time:.2f}x")

    print("\nKEY FINDINGS")
    print("-" * 40)
    if best_k is not None:
        print(f"  Best improvement at k={best_k}: +{best_improvement:.4f} (+{best_improvement/base_pass_k.get(f'pass@{best_k}',1)*100:.1f}%)")

    # Check if tree is better across all k
    all_better = all(tree_pass_k.get(f'pass@{k}', 0) > base_pass_k.get(f'pass@{k}', 0) for k in range(1, n + 1))
    if all_better:
        print("  Tree Decoding is better across all pass@k!")

    # Average improvement
    avg_abs_imp = np.mean([tree_pass_k.get(f'pass@{k}', 0) - base_pass_k.get(f'pass@{k}', 0) for k in range(1, n + 1)])
    avg_rel_imp = np.mean([(tree_pass_k.get(f'pass@{k}', 0) - base_pass_k.get(f'pass@{k}', 0)) / base_pass_k.get(f'pass@{k}', 1) * 100 for k in range(1, n + 1)])
    print(f"  Average absolute improvement: {avg_abs_imp:+.4f}")
    print(f"  Average relative improvement: {avg_rel_imp:+.1f}%")


def analyze_generations(results: Dict[str, Any]):
    """Analyze generation details."""
    if 'generations' not in results:
        return

    gens = results['generations']
    base_results = gens.get('base_grpo', {}).get('results', [])
    tree_results = gens.get('tree_decoding', {}).get('results', [])
    examples = gens.get('examples', [])

    if not base_results or not tree_results:
        return

    print("\nGENERATION ANALYSIS")
    print("-" * 40)

    # Find problems where tree did better
    tree_better = []
    base_better = []
    same = []

    for idx, (base_res, tree_res, ex) in enumerate(zip(base_results, tree_results, examples)):
        base_has_correct = any(base_res)
        tree_has_correct = any(tree_res)

        if tree_has_correct and not base_has_correct:
            tree_better.append((idx, ex))
        elif base_has_correct and not tree_has_correct:
            base_better.append((idx, ex))
        else:
            same.append((idx, ex, base_has_correct))

    print(f"  Problems where Tree succeeded but Base failed: {len(tree_better)}")
    print(f"  Problems where Base succeeded but Tree failed: {len(base_better)}")
    print(f"  Problems with same outcome:                  {len(same)}")

    # Per-position correctness
    print("\nPER-POSITION CORRECTNESS")
    print("-" * 40)

    n = len(base_results[0]) if base_results else 8

    print(f"\n{'Position':<10} {'Base GRPO':<12} {'Tree Decoding':<12}")
    print("-" * 34)

    for pos in range(n):
        base_correct = sum(1 for res in base_results if pos < len(res) and res[pos])
        tree_correct = sum(1 for res in tree_results if pos < len(res) and res[pos])
        base_rate = base_correct / len(base_results) if base_results else 0
        tree_rate = tree_correct / len(tree_results) if tree_results else 0
        print(f"{pos + 1:<10} {base_rate:<12.4f} {tree_rate:<12.4f}")


def create_comparison_table(results: Dict[str, Any], output_dir: Path):
    """Create detailed comparison tables."""
    if 'summary' not in results:
        return

    summary = results['summary']
    base = summary['base_grpo']
    tree = summary['tree_decoding']
    config = summary['config']

    n = config.get('n', 8)

    # Create DataFrame
    rows = []
    for k in range(1, n + 1):
        base_score = base['pass_k'].get(f'pass@{k}', 0)
        tree_score = tree['pass_k'].get(f'pass@{k}', 0)
        rows.append({
            'k': k,
            'base_grpo': base_score,
            'tree_decoding': tree_score,
            'absolute_improvement': tree_score - base_score,
            'relative_improvement': (tree_score - base_score) / base_score * 100 if base_score > 0 else 0,
        })

    df = pd.DataFrame(rows)

    # Save detailed CSV
    detailed_csv = output_dir / "results_detailed.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"\nSaved detailed results to: {detailed_csv}")

    # Print markdown table
    print("\nMARKDOWN SUMMARY TABLE")
    print("-" * 40)
    print("\n| k | Base GRPO | Tree Decoding | Abs Improv | Rel Improv |")
    print("|---|-----------|---------------|------------|------------|")
    for _, row in df.iterrows():
        print(f"| {row['k']} | {row['base_grpo']:.4f} | {row['tree_decoding']:.4f} | {row['absolute_improvement']:+.4f} | {row['relative_improvement']:+.1f}% |")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Tree Decoding vs Base GRPO results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./tree_decoding_results",
        help="Directory with results"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)

    print_detailed_summary(results)
    analyze_generations(results)
    create_comparison_table(results, results_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

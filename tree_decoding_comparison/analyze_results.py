#!/usr/bin/env python3
"""Analyze four-way tree-decoding comparison results (and legacy two-way runs)."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "base_grpo": "Base GRPO",
    "random_tree": "Random tree",
    "entropy_only_tree": "Entropy-only tree",
    "entropy_waad_tree": "Entropy+WAAD tree",
    # Backward compatibility for schema_version=1 result directories.
    "tree_decoding": "Tree decoding (legacy)",
}
SCHEMA_V2_METHODS = {
    "base_grpo",
    "random_tree",
    "entropy_only_tree",
    "entropy_waad_tree",
}


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all result artifacts present in a directory."""
    results = {}
    for key, filename in (
        ("summary", "results_summary.json"),
        ("generations", "generations.json"),
    ):
        path = results_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                results[key] = json.load(handle)
    return results


def available_methods(payload: Dict[str, Any]) -> List[str]:
    """Return methods in stable display order."""
    return [key for key in METHOD_LABELS if key in payload]


def validate_summary_schema(summary: Dict[str, Any]) -> None:
    """Reject incomplete v2 artifacts instead of plotting missing data as 0."""
    if int(summary.get("schema_version", 1)) < 2:
        return
    missing_methods = SCHEMA_V2_METHODS - set(summary)
    if missing_methods:
        raise ValueError(
            f"Schema v2 summary is missing methods: {sorted(missing_methods)}")
    n = int(summary.get("config", {}).get("n", 0))
    if n < 1:
        raise ValueError("Schema v2 summary has an invalid candidate budget n.")
    for method in SCHEMA_V2_METHODS:
        pass_k = summary[method].get("pass_k", {})
        missing_metrics = [
            f"pass@{k}" for k in range(1, n + 1)
            if f"pass@{k}" not in pass_k
        ]
        if missing_metrics:
            raise ValueError(
                f"Schema v2 method {method} is missing metrics: "
                f"{missing_metrics}")


def validate_generation_schema(generations: Dict[str, Any]) -> None:
    """Ensure every v2 method contains one result row per example."""
    if int(generations.get("schema_version", 1)) < 2:
        return
    missing_methods = SCHEMA_V2_METHODS - set(generations)
    if missing_methods:
        raise ValueError(
            f"Schema v2 generations are missing methods: "
            f"{sorted(missing_methods)}")
    candidate_budget = int(generations.get("candidate_budget_n", 0))
    if candidate_budget < 1:
        raise ValueError(
            "Schema v2 generations have an invalid candidate budget n.")
    expected_rows = len(generations.get("examples", []))
    for method in SCHEMA_V2_METHODS:
        result_rows = generations[method].get("results", [])
        candidate_rows = generations[method].get("generations", [])
        metadata_rows = generations[method].get("candidate_metadata", [])
        if not (
            len(result_rows) == len(candidate_rows) == len(metadata_rows)
            == expected_rows
        ):
            raise ValueError(
                f"Schema v2 method {method} has inconsistent row counts: "
                f"results={len(result_rows)}, candidates={len(candidate_rows)}, "
                f"metadata={len(metadata_rows)}, examples={expected_rows}.")
        for row_idx, (results, candidates, metadata) in enumerate(zip(
                result_rows, candidate_rows, metadata_rows)):
            source_count = len(metadata.get("candidate_sources", []))
            total_count = int(metadata.get("total_count", -1))
            if not (
                len(results) == len(candidates) == source_count
                == total_count == candidate_budget
            ):
                raise ValueError(
                    f"Schema v2 method {method} row {row_idx} violates "
                    f"candidate budget {candidate_budget}: results="
                    f"{len(results)}, candidates={len(candidates)}, "
                    f"sources={source_count}, total_count={total_count}.")


def validate_artifact_pair(results: Dict[str, Any]) -> None:
    """Reject schema-v2 summary/generation files from different runs."""
    summary = results.get("summary")
    generations = results.get("generations")
    if not summary or not generations:
        return
    summary_version = int(summary.get("schema_version", 1))
    generations_version = int(generations.get("schema_version", 1))
    if summary_version < 2 and generations_version < 2:
        return
    if (summary_version < 2) != (generations_version < 2):
        raise ValueError(
            "results_summary.json and generations.json use incompatible "
            f"schema versions: {summary_version} and {generations_version}.")

    summary_run_id = summary.get("run_id")
    generations_run_id = generations.get("run_id")
    if not summary_run_id or not generations_run_id:
        raise ValueError(
            "Schema v2 artifacts must both contain a non-empty run_id.")
    if summary_run_id != generations_run_id:
        raise ValueError(
            "results_summary.json and generations.json belong to different "
            "experiment runs.")

    summary_budget = int(summary.get("config", {}).get("n", 0))
    generation_budget = int(generations.get("candidate_budget_n", 0))
    if summary_budget != generation_budget:
        raise ValueError(
            "Summary and generation candidate budgets differ: "
            f"{summary_budget} != {generation_budget}.")


def print_detailed_summary(results: Dict[str, Any]) -> None:
    """Print configuration, pass@k, timing, and budget composition."""
    summary = results.get("summary")
    if not summary:
        print("No summary found")
        return
    validate_summary_schema(summary)

    methods = available_methods(summary)
    if "base_grpo" not in methods:
        raise ValueError("results_summary.json does not contain base_grpo.")
    config = summary.get("config", {})
    n = int(config.get("n", 8))

    print("=" * 100)
    print("TREE DECODING vs BASE GRPO - DETAILED ANALYSIS")
    print("=" * 100)
    print(f"Pass@k definition: {summary.get('pass_at_k_definition', 'legacy prefix')}")
    for key in (
        "model", "num_samples", "n", "branching_factor", "max_tree_depth",
        "min_seg_length", "random_branch_probability", "entropy_threshold",
        "tau_importance", "temperature", "top_p", "top_k", "max_tokens",
        "seed", "vllm_module",
    ):
        if key in config:
            print(f"  {key}: {config[key]}")

    width = 23
    print("\nPASS@K COMPARISON")
    print(f"{'Metric':<12}" + "".join(
        f"{METHOD_LABELS[key]:<{width}}" for key in methods))
    print("-" * (12 + width * len(methods)))
    for k in range(1, n + 1):
        metric = f"pass@{k}"
        print(f"{metric:<12}" + "".join(
            f"{summary[key].get('pass_k', {}).get(metric, 0):<{width}.4f}"
            for key in methods))

    print("\nTIMING AND TREE/FILLER COMPOSITION")
    base_time = float(summary["base_grpo"].get("time", 0))
    for key in methods:
        method = summary[key]
        elapsed = float(method.get("time", 0))
        budget = method.get("budget", {})
        ratio = elapsed / base_time if base_time > 0 else float("nan")
        print(
            f"  {METHOD_LABELS[key]:<24} {elapsed:>10.2f}s "
            f"({ratio:>6.2f}x base), mean leaves="
            f"{budget.get('tree_leaf_count_mean', 0):.2f}, mean fillers="
            f"{budget.get('filler_count_mean', 0):.2f}")

    print("\nDELTA FROM BASE")
    base_pass_k = summary["base_grpo"].get("pass_k", {})
    for key in methods:
        if key == "base_grpo":
            continue
        deltas = [
            summary[key].get("pass_k", {}).get(f"pass@{k}", 0)
            - base_pass_k.get(f"pass@{k}", 0)
            for k in range(1, n + 1)
        ]
        best_k = int(np.argmax(deltas)) + 1
        print(
            f"  {METHOD_LABELS[key]:<24} mean={np.mean(deltas):+.4f}, "
            f"best=pass@{best_k} {deltas[best_k - 1]:+.4f}")


def analyze_generations(results: Dict[str, Any]) -> None:
    """Compare each candidate pool with base without relying on list position."""
    generations = results.get("generations")
    if not generations or "base_grpo" not in generations:
        return
    validate_generation_schema(generations)

    methods = available_methods(generations)
    base_results = generations["base_grpo"].get("results", [])
    if not base_results:
        return

    print("\nGENERATION-POOL ANALYSIS")
    base_success = [any(row) for row in base_results]
    for key in methods:
        rows = generations[key].get("results", [])
        if not rows:
            continue
        pool_success = [any(row) for row in rows]
        candidate_accuracy = float(np.mean([
            float(np.mean(row)) if row else 0.0 for row in rows
        ]))
        if key == "base_grpo":
            print(
                f"  {METHOD_LABELS[key]:<24} solved={sum(pool_success):>4}/"
                f"{len(pool_success)}, candidate accuracy="
                f"{candidate_accuracy:.4f}")
            continue

        method_only = sum(
            method_ok and not base_ok
            for method_ok, base_ok in zip(pool_success, base_success))
        base_only = sum(
            base_ok and not method_ok
            for method_ok, base_ok in zip(pool_success, base_success))
        print(
            f"  {METHOD_LABELS[key]:<24} solved={sum(pool_success):>4}/"
            f"{len(pool_success)}, candidate accuracy={candidate_accuracy:.4f}, "
            f"method-only={method_only}, base-only={base_only}")


def create_comparison_table(
    results: Dict[str, Any], output_dir: Path,
) -> None:
    """Write a detailed dynamic CSV and print a Markdown table."""
    summary = results.get("summary")
    if not summary:
        return
    validate_summary_schema(summary)
    methods = available_methods(summary)
    config = summary.get("config", {})
    n = int(config.get("n", 8))

    rows = []
    for k in range(1, n + 1):
        metric = f"pass@{k}"
        row = {"k": k}
        for key in methods:
            row[key] = summary[key].get("pass_k", {}).get(metric, 0)
            if key != "base_grpo":
                row[f"{key}_minus_base"] = row[key] - row["base_grpo"]
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    detailed_csv = output_dir / "results_detailed.csv"
    dataframe.to_csv(detailed_csv, index=False)
    print(f"\nSaved detailed results to: {detailed_csv}")

    print("\nMARKDOWN SUMMARY TABLE")
    headers = ["k"] + [METHOD_LABELS[key] for key in methods]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [str(row["k"])] + [f"{row[key]:.4f}" for key in methods]
        print("| " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Tree Decoding vs Base GRPO results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./tree_decoding_results",
        help="Directory with results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results = load_results(results_dir)
    validate_artifact_pair(results)
    print_detailed_summary(results)
    analyze_generations(results)
    create_comparison_table(results, results_dir)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
End-to-end comparison experiment between Tree Decoding and Base GRPO sampling.

This script:
1. Loads and samples 500 unique queries from dapo-math dataset
2. Uses both Tree Decoding and Base GRPO sampling to generate responses
3. Ensures each method produces exactly 8 sequences per query
4. Evaluates pass@1 to pass@8 for both methods
5. Outputs comparison results
"""

import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add vllm to path
sys.path.insert(0, '/Users/weilongxuan/codes/vllm')

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import TreeSearchParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vllm not available")


@dataclass
class MathExample:
    """Data class for a math problem example."""
    problem: str
    ground_truth: str
    prompt: str
    idx: int


def load_dapo_math_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load dapo-math dataset from parquet file.

    Args:
        dataset_path: Path to dapo-math-17k.parquet

    Returns:
        DataFrame with dataset
    """
    import pyarrow.parquet as pq
    table = pq.read_table(dataset_path)
    df = table.to_pandas()
    print(f"Loaded dataset with {len(df)} rows")
    return df


def extract_unique_queries(df: pd.DataFrame, num_samples: int = 500) -> List[MathExample]:
    """
    Extract unique queries from dataset.

    Args:
        df: Input dataframe
        num_samples: Number of unique samples to extract

    Returns:
        List of MathExample objects
    """
    # Find unique problems
    if 'prompt' in df.columns:
        # Try to extract problem from prompt
        unique_df = df.drop_duplicates(subset=['prompt'])
    elif 'problem' in df.columns:
        unique_df = df.drop_duplicates(subset=['problem'])
    else:
        # Assume first column is the problem
        unique_df = df.drop_duplicates()

    print(f"Found {len(unique_df)} unique problems")

    # Sample if needed
    if len(unique_df) > num_samples:
        unique_df = unique_df.sample(n=num_samples, random_state=42)

    examples = []
    for idx, row in unique_df.iterrows():
        # Extract problem and ground truth based on available columns
        if 'prompt' in df.columns and 'reward_model' in df.columns:
            # Format from verl's preprocessed data
            prompt = row['prompt']
            if isinstance(prompt, list) and len(prompt) > 0:
                prompt_text = prompt[0].get('content', '')
            else:
                prompt_text = str(prompt)
            ground_truth = row.get('reward_model', {}).get('ground_truth', '')
            problem = prompt_text
        elif 'problem' in df.columns and 'solution' in df.columns:
            # Original MATH dataset format
            problem = row['problem']
            solution = row['solution']
            ground_truth = extract_answer_from_solution(solution)
            # Add instruction following prompt
            prompt_text = problem + " Let's think step by step and output the final answer within \\boxed{}."
        else:
            # Try to infer from columns
            problem = str(row.iloc[0])
            prompt_text = problem
            ground_truth = str(row.iloc[1]) if len(row) > 1 else ''

        examples.append(MathExample(
            problem=problem,
            ground_truth=ground_truth,
            prompt=prompt_text,
            idx=len(examples)
        ))

    print(f"Extracted {len(examples)} unique examples")
    return examples


def extract_answer_from_solution(solution: str) -> str:
    """Extract boxed answer from solution string."""
    try:
        idx = solution.rfind("\\boxed")
        if idx < 0:
            idx = solution.rfind("\\fbox")
            if idx < 0:
                return solution.strip()

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(solution):
            if solution[i] == "{":
                num_left_braces_open += 1
            if solution[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return solution.strip()
        return solution[idx:right_brace_idx + 1].replace("\\boxed", "").replace("\\fbox", "").strip("{}")
    except Exception:
        return solution.strip()


# ========== Math Grading Functions ==========
# These are adapted from verl's math_reward module


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract last boxed content from string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove boxed wrapper from string."""
    left = "\\boxed{"
    try:
        if s[:len(left)] == left and s[-1] == "}":
            return s[len(left):-1]
    except Exception:
        pass
    return s


def extract_answer(generation: str) -> Optional[str]:
    """Extract answer from generation."""
    if "\\boxed" in generation or "\\fbox" in generation:
        boxed = last_boxed_only_string(generation)
        if boxed:
            return remove_boxed(boxed)
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize answer for comparison."""
    if answer is None:
        return None
    answer = answer.strip()

    # Remove units and common patterns
    import re
    answer = re.sub(r'\\text\{.*?\}$', '', answer).strip()
    answer = re.sub(r'\\text\{', '', answer)
    answer = answer.replace('}', '')
    answer = answer.replace('{', '')
    answer = answer.replace('$', '')

    # Try to extract numerical value
    try:
        # Check if it's a number
        val = float(answer.replace(',', ''))
        if val.is_integer():
            return str(int(val))
        return str(val)
    except (ValueError, TypeError):
        pass

    return answer


def grade_answer(model_answer: Optional[str], ground_truth: str) -> bool:
    """
    Grade if model answer matches ground truth.

    Args:
        model_answer: Extracted answer from model
        ground_truth: Ground truth answer

    Returns:
        True if correct, False otherwise
    """
    if model_answer is None:
        return False

    # Normalize both answers
    norm_model = normalize_answer(model_answer)
    norm_gt = normalize_answer(ground_truth)

    if norm_model is None or norm_gt is None:
        return False

    # Direct string match
    if norm_model == norm_gt:
        return True

    # Try numerical comparison
    try:
        model_val = float(norm_model.replace(',', ''))
        gt_val = float(norm_gt.replace(',', ''))
        if abs(model_val - gt_val) < 1e-4:
            return True
    except (ValueError, TypeError):
        pass

    # Try symbolic comparison as fallback
    try:
        import sympy
        model_expr = sympy.sympify(norm_model)
        gt_expr = sympy.sympify(norm_gt)
        if sympy.simplify(model_expr - gt_expr) == 0:
            return True
    except Exception:
        pass

    return False


def evaluate_generations(generations: List[str], ground_truth: str) -> List[bool]:
    """
    Evaluate a list of generations against ground truth.

    Args:
        generations: List of generated responses
        ground_truth: Ground truth answer

    Returns:
        List of booleans indicating correctness
    """
    results = []
    for gen in generations:
        answer = extract_answer(gen)
        correct = grade_answer(answer, ground_truth)
        results.append(correct)
    return results


# ========== Threshold Calibration ==========

def collect_threshold_stats(
    llm: LLM,
    prompts: List[str],
    n: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 200,
    quantile: float = 0.8,
) -> Tuple[float, float]:
    """
    Collect entropy and importance stats using vllm's collect_threshold_stats mode,
    then compute quantiles as thresholds.

    Args:
        llm: vLLM instance
        prompts: List of prompts for calibration
        n: Number of sequences per prompt for calibration
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Max tokens for calibration (shorter than full generation)
        quantile: Quantile to use for threshold (0.8 = 80th percentile)

    Returns:
        Tuple of (entropy_threshold, tau_importance)
    """
    print("\nCollecting threshold stats...")

    all_entropies = []
    all_importances = []

    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        collect_threshold_stats=True
    )

    # Process prompts in one batch
    outputs = llm.generate(prompts, params)

    for output in outputs:
        for i in range(len(output.outputs)):
            result = output.outputs[i]
            if hasattr(result, 'entropy_list') and result.entropy_list:
                all_entropies.extend(result.entropy_list)
            if hasattr(result, 'importance_list') and result.importance_list:
                all_importances.extend([x for x in result.importance_list if x is not None])

    # Compute quantiles
    entropy_threshold = 1.0  # default
    tau_importance = 0.0   # default

    if all_entropies:
        entropy_threshold = float(np.quantile(all_entropies, quantile))
        print(f"  Entropy stats: {len(all_entropies)} tokens")
        print(f"  Entropy range: [{min(all_entropies):.4f}, {max(all_entropies):.4f}]")
        print(f"  Entropy {int(quantile*100)}th percentile: {entropy_threshold:.4f}")
    else:
        print("  Warning: No entropy data collected, using default entropy_threshold")

    if all_importances:
        tau_importance = float(np.quantile(all_importances, quantile))
        print(f"  Importance stats: {len(all_importances)} tokens")
        print(f"  Importance range: [{min(all_importances):.4f}, {max(all_importances):.4f}]")
        print(f"  Importance {int(quantile*100)}th percentile: {tau_importance:.4f}")
    else:
        print("  Warning: No importance data collected, using default tau_importance")

    return entropy_threshold, tau_importance


# ========== Pass@k Calculation ==========

def compute_pass_at_k(results_per_example: List[List[bool]], k: int) -> float:
    """
    Compute pass@k.

    Args:
        results_per_example: List where each element is a list of booleans
                            indicating correctness for that example's generations
        k: k for pass@k

    Returns:
        pass@k score
    """
    total = 0
    correct = 0

    for results in results_per_example:
        if len(results) == 0:
            continue
        total += 1
        # Take first min(k, len(results)) results and see if any is correct
        n = min(k, len(results))
        if any(results[:n]):
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_all_pass_k(results_per_example: List[List[bool]], max_k: int = 8) -> Dict[str, float]:
    """
    Compute pass@1 to pass@max_k.

    Args:
        results_per_example: List where each element is a list of booleans
        max_k: Maximum k to compute

    Returns:
        Dictionary with pass@k scores
    """
    pass_k_scores = {}
    for k in range(1, max_k + 1):
        pass_k_scores[f"pass@{k}"] = compute_pass_at_k(results_per_example, k)
    return pass_k_scores


# ========== Generation with vLLM ==========

def generate_base_grpo(
    llm: LLM,
    prompts: List[str],
    n: int = 8,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 4096,
) -> List[List[str]]:
    """
    Generate using base GRPO sampling (independent sampling).

    Default parameters match verl's RolloutConfig:
    - temperature: 1.0
    - top_p: 1.0
    - top_k: -1

    Args:
        llm: vLLM instance
        prompts: List of prompts
        n: Number of sequences per prompt
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Max tokens to generate

    Returns:
        List where each element is a list of n generations for that prompt
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        top_p=top_p,
        top_k=top_k,
    )

    outputs = llm.generate(prompts, sampling_params)

    all_generations = []
    for output in outputs:
        gens = [o.text for o in output.outputs]
        all_generations.append(gens)

    return all_generations


def generate_tree_decoding(
    llm: LLM,
    prompts: List[str],
    branching_factor: int = 2,
    max_tree_depth: int = 3,
    target_n: int = 8,
    entropy_threshold: float = 1.0,
    tau_importance: Optional[float] = 0.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 4096,
) -> List[List[str]]:
    """
    Generate using Tree Decoding.

    Default parameters match verl's TreeSearchConfig:
    - branching_factor: 2
    - max_tree_depth: 3
    - entropy_threshold: 1.0
    - tau_importance: 0.0

    If tree doesn't produce enough leaves, fills remaining with base sampling.

    Args:
        llm: vLLM instance
        prompts: List of prompts
        branching_factor: Branching factor for tree
        max_tree_depth: Max depth for tree
        target_n: Target number of sequences per prompt
        entropy_threshold: Entropy threshold for branching
        tau_importance: Tau for importance sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Max tokens to generate

    Returns:
        List where each element is a list of target_n generations for that prompt
    """
    tree_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=entropy_threshold,
            branching_factor=branching_factor,
            max_tree_depth=max_tree_depth,
            tau_importance=tau_importance,
        )
    )

    # First pass: get tree decoding outputs
    outputs = llm.generate(prompts, tree_params)

    all_generations = []
    prompts_needing_filler = []
    filler_start_indices = []

    # Extract leaf nodes first
    for idx, output in enumerate(outputs):
        leaf_generations = []

        # Build a map from seq_id to output object for parent lookup
        seq_map = {}
        for o in output.outputs:
            if hasattr(o, 'seq_id') and o.seq_id is not None:
                seq_map[o.seq_id] = o

        for o in output.outputs:
            if hasattr(o, 'is_leaf') and o.is_leaf:
                # Reconstruct full sequence by traversing from leaf to root
                full_text = None
                if hasattr(o, 'parent_seq_id'):
                    # Use tree_text segments and backtrack through parents
                    path_texts = []
                    current = o
                    while current is not None:
                        if hasattr(current, 'tree_text') and current.tree_text:
                            path_texts.append(current.tree_text)
                        if hasattr(current, 'parent_seq_id') and current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                            current = seq_map[current.parent_seq_id]
                        else:
                            current = None

                    # The path gives leaf to root, so we reverse it
                    if path_texts:
                        full_text = "".join(reversed(path_texts))

                # Fall back to direct text if backtracking didn't work
                if full_text is None or len(full_text.strip()) == 0:
                    if hasattr(o, 'text') and o.text:
                        full_text = o.text
                    elif hasattr(o, 'tree_text') and o.tree_text:
                        full_text = o.tree_text

                if full_text:
                    leaf_generations.append(full_text)

            elif not hasattr(o, 'is_leaf'):
                # If is_leaf attribute not available, try backtracking or take text
                full_text = None
                if hasattr(o, 'parent_seq_id'):
                    # Try backtracking
                    path_texts = []
                    current = o
                    while current is not None:
                        if hasattr(current, 'tree_text') and current.tree_text:
                            path_texts.append(current.tree_text)
                        if hasattr(current, 'parent_seq_id') and current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                            current = seq_map[current.parent_seq_id]
                        else:
                            current = None

                    if path_texts:
                        full_text = "".join(reversed(path_texts))

                if full_text is None or len(full_text.strip()) == 0:
                    if hasattr(o, 'text') and o.text:
                        full_text = o.text
                    elif hasattr(o, 'tree_text') and o.tree_text:
                        full_text = o.tree_text

                if full_text:
                    leaf_generations.append(full_text)

        # Deduplicate while preserving order
        seen = set()
        unique_leaves = []
        for gen in leaf_generations:
            if gen not in seen:
                seen.add(gen)
                unique_leaves.append(gen)
        leaf_generations = unique_leaves

        if len(leaf_generations) >= target_n:
            # Take first target_n
            all_generations.append(leaf_generations[:target_n])
        else:
            # Need to fill remaining with base sampling
            all_generations.append(leaf_generations.copy())
            prompts_needing_filler.append(prompts[idx])
            filler_start_indices.append(idx)

    # Second pass: fill in missing with base sampling if needed
    if prompts_needing_filler:
        print(f"Filling missing sequences for {len(prompts_needing_filler)} prompts...")
        needed_per_prompt = [target_n - len(all_generations[idx]) for idx in filler_start_indices]

        # Collect all filler prompts with their needed counts
        all_filler_prompts = []
        for prompt, needed in zip(prompts_needing_filler, needed_per_prompt):
            all_filler_prompts.extend([prompt] * needed)

        if all_filler_prompts:
            # Generate in batch (each with n=1)
            filler_sampling_params = SamplingParams(
                temperature=temperature,  # Slightly higher temperature for diversity
                max_tokens=max_tokens,
                n=1,
                top_p=top_p,
                top_k=top_k,
            )

            filler_outputs = llm.generate(all_filler_prompts, filler_sampling_params)

            # Distribute back to original prompts
            filler_idx = 0
            for orig_idx, needed in zip(filler_start_indices, needed_per_prompt):
                for _ in range(needed):
                    if filler_idx < len(filler_outputs):
                        text = filler_outputs[filler_idx].outputs[0].text
                        all_generations[orig_idx].append(text)
                    filler_idx += 1

    # Ensure all have exactly target_n (in case anything went wrong)
    for i in range(len(all_generations)):
        while len(all_generations[i]) < target_n:
            # Duplicate last or add empty
            if all_generations[i]:
                all_generations[i].append(all_generations[i][-1])
            else:
                all_generations[i].append("")
        all_generations[i] = all_generations[i][:target_n]

    return all_generations


# ========== Main Experiment ==========

def run_experiment(args):
    """Run the full comparison experiment."""
    print("=" * 80)
    print("Tree Decoding vs Base GRPO Comparison Experiment")
    print("=" * 80)

    # Check dataset path
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        # Try to find from environment variable
        ray_data_home = os.environ.get('RAY_DATA_HOME', '')
        if ray_data_home:
            alt_path = os.path.join(ray_data_home, 'dapo-math-17k.parquet')
            if os.path.exists(alt_path):
                dataset_path = alt_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            f"Please set RAY_DATA_HOME environment variable or provide correct path."
        )

    print(f"\n1. Loading dataset from: {dataset_path}")
    df = load_dapo_math_dataset(dataset_path)

    print(f"\n2. Extracting {args.num_samples} unique queries...")
    examples = extract_unique_queries(df, num_samples=args.num_samples)

    # Save examples for reference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_file = output_dir / "sampled_examples.json"
    with open(examples_file, 'w') as f:
        json.dump([{
            'idx': ex.idx,
            'problem': ex.problem,
            'prompt': ex.prompt,
            'ground_truth': ex.ground_truth
        } for ex in examples], f, indent=2)
    print(f"   Saved sampled examples to: {examples_file}")

    # Initialize LLM
    print(f"\n3. Initializing vLLM with model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )

    prompts = [ex.prompt for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]

    # Calibrate thresholds if auto-calibration is enabled
    entropy_threshold = args.entropy_threshold
    tau_importance = args.tau_importance

    if args.auto_calibrate_thresholds:
        print(f"\n3.5. Auto-calibrating thresholds (using {args.calibration_quantile*100}th percentile)...")
        entropy_threshold, tau_importance = collect_threshold_stats(
            llm=llm,
            prompts=prompts,
            n=args.calibration_n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.calibration_max_tokens,
            quantile=args.calibration_quantile,
        )
        print(f"\nCalibrated thresholds:")
        print(f"  entropy_threshold: {entropy_threshold:.4f}")
        print(f"  tau_importance:    {tau_importance:.4f}")

    # Run Base GRPO sampling
    print(f"\n4. Generating with Base GRPO (n={args.n})...")
    start_time = time.time()
    base_generations = generate_base_grpo(
        llm=llm,
        prompts=prompts,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    base_time = time.time() - start_time
    print(f"   Base GRPO time: {base_time:.2f}s")

    # Evaluate Base GRPO
    print("\n5. Evaluating Base GRPO generations...")
    base_results = []
    for gens, gt in zip(base_generations, ground_truths):
        results = evaluate_generations(gens, gt)
        base_results.append(results)

    base_pass_k = compute_all_pass_k(base_results, max_k=args.n)
    print("   Base GRPO results:")
    for k, v in base_pass_k.items():
        print(f"     {k}: {v:.4f}")

    # Run Tree Decoding
    print(f"\n6. Generating with Tree Decoding (branching_factor={args.branching_factor}, "
          f"max_depth={args.max_tree_depth}, entropy_threshold={entropy_threshold:.4f}, "
          f"tau_importance={tau_importance:.4f})...")
    start_time = time.time()
    tree_generations = generate_tree_decoding(
        llm=llm,
        prompts=prompts,
        branching_factor=args.branching_factor,
        max_tree_depth=args.max_tree_depth,
        target_n=args.n,
        entropy_threshold=entropy_threshold,
        tau_importance=tau_importance,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    tree_time = time.time() - start_time
    print(f"   Tree Decoding time: {tree_time:.2f}s")

    # Evaluate Tree Decoding
    print("\n7. Evaluating Tree Decoding generations...")
    tree_results = []
    for gens, gt in zip(tree_generations, ground_truths):
        results = evaluate_generations(gens, gt)
        tree_results.append(results)

    tree_pass_k = compute_all_pass_k(tree_results, max_k=args.n)
    print("   Tree Decoding results:")
    for k, v in tree_pass_k.items():
        print(f"     {k}: {v:.4f}")

    # Save generations
    generations_file = output_dir / "generations.json"
    generations_data = {
        'examples': [{
            'idx': ex.idx,
            'problem': ex.problem,
            'prompt': ex.prompt,
            'ground_truth': ex.ground_truth,
        } for ex in examples],
        'base_grpo': {
            'generations': base_generations,
            'results': base_results,
        },
        'tree_decoding': {
            'generations': tree_generations,
            'results': tree_results,
        }
    }
    with open(generations_file, 'w') as f:
        json.dump(generations_data, f, indent=2)
    print(f"\n8. Saved generations to: {generations_file}")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<15} {'Base GRPO':<15} {'Tree Decoding':<15} {'Improvement':<15}")
    print("-" * 60)

    for k in range(1, args.n + 1):
        base_score = base_pass_k[f"pass@{k}"]
        tree_score = tree_pass_k[f"pass@{k}"]
        improvement = tree_score - base_score
        improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0
        print(f"{f'pass@{k}':<15} {base_score:<15.4f} {tree_score:<15.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")

    print("-" * 60)
    print(f"{'Time (s)':<15} {base_time:<15.2f} {tree_time:<15.2f} {tree_time-base_time:+.2f}")

    # Save summary
    summary = {
        'base_grpo': {
            'pass_k': base_pass_k,
            'time': base_time,
        },
        'tree_decoding': {
            'pass_k': tree_pass_k,
            'time': tree_time,
        },
        'config': {
            'model': args.model_path,
            'num_samples': args.num_samples,
            'n': args.n,
            'branching_factor': args.branching_factor,
            'max_tree_depth': args.max_tree_depth,
            'entropy_threshold': entropy_threshold,
            'tau_importance': tau_importance,
            'auto_calibrate_thresholds': args.auto_calibrate_thresholds,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
        }
    }

    summary_file = output_dir / "results_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results summary to: {summary_file}")

    # Also save as CSV for easier analysis
    csv_data = []
    for k in range(1, args.n + 1):
        csv_data.append({
            'k': k,
            'base_grpo': base_pass_k[f"pass@{k}"],
            'tree_decoding': tree_pass_k[f"pass@{k}"],
        })
    csv_file = output_dir / "results_summary.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    print(f"Saved CSV summary to: {csv_file}")

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Tree Decoding vs Base GRPO Comparison Experiment"
    )

    # Dataset
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="${RAY_DATA_HOME}/data/dapo-math-17k.parquet",
        help="Path to dapo-math dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of unique queries to sample"
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length"
    )

    # Sampling
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Number of sequences per prompt (total sequences per query)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (verl default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (verl default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (verl default: -1)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate"
    )

    # Tree Decoding
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=2,
        help="Tree branching factor (verl default: 2)"
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=3,
        help="Max tree depth (verl default: 3)"
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=1.0,
        help="Entropy threshold for branching (verl default: 1.0)"
    )
    parser.add_argument(
        "--tau-importance",
        type=float,
        default=0.0,
        help="Tau for importance sampling (verl default: 0.0)"
    )

    # Threshold Auto-calibration
    parser.add_argument(
        "--auto-calibrate-thresholds",
        action="store_true",
        help="Auto-calibrate entropy_threshold and tau_importance using collect_threshold_stats mode"
    )
    parser.add_argument(
        "--calibration-n",
        type=int,
        default=5,
        help="Number of sequences per prompt for threshold calibration"
    )
    parser.add_argument(
        "--calibration-max-tokens",
        type=int,
        default=200,
        help="Max tokens for threshold calibration (shorter than full generation)"
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.8,
        help="Quantile to use for threshold (0.8 = 80th percentile)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tree_decoding_results",
        help="Output directory"
    )

    args = parser.parse_args()

    # Expand environment variables in path
    if '${RAY_DATA_HOME}' in args.dataset_path:
        ray_data_home = os.environ.get('RAY_DATA_HOME', '')
        args.dataset_path = args.dataset_path.replace('${RAY_DATA_HOME}', ray_data_home)

    run_experiment(args)


if __name__ == "__main__":
    main()

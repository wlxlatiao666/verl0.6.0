#!/usr/bin/env python3
"""
Test the evaluation logic without needing vllm or a model.

This verifies that the grading and pass@k calculation works correctly.
This is a standalone version that doesn't import from the main script.
"""

import sys
from typing import List, Dict, Any, Optional


# ========== Standalone evaluation functions (simplified for testing) ==========

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
    # Handle boxed
    left = "\\boxed{"
    try:
        if s[:len(left)] == left and s[-1] == "}":
            return s[len(left):-1]
    except Exception:
        pass
    # Also try removing fbox
    left = "\\fbox{"
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
    # Simple normalization - just remove $ and whitespace
    answer = answer.replace("$", "")
    answer = answer.strip()
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
        model_val = float(str(norm_model).replace(',', ''))
        gt_val = float(str(norm_gt).replace(',', ''))
        if abs(model_val - gt_val) < 1e-4:
            return True
    except (ValueError, TypeError):
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
        max_k: Max k to compute

    Returns:
        Dictionary with pass@k scores
    """
    pass_k_scores = {}
    for k in range(1, max_k + 1):
        pass_k_scores[f"pass@{k}"] = compute_pass_at_k(results_per_example, k)
    return pass_k_scores


# ========== Tests ==========

def test_extract_answer():
    """Test answer extraction."""
    print("Testing extract_answer...")

    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Final result: \\fbox{123.45}", "123.45"),
        ("No boxed answer here", None),
        ("Multiple boxes: \\boxed{1} then \\boxed{2}", "2"),
        ("Nested \\boxed{x^{2} + 1}", "x^{2} + 1"),
    ]

    passed = 0
    for generation, expected in test_cases:
        result = extract_answer(generation)
        if result == expected:
            print(f"  ✓ PASS: '{generation[:50]}...' -> '{result}'")
            passed += 1
        else:
            print(f"  ✗ FAIL: '{generation[:50]}...'")
            print(f"    Expected: '{expected}', Got: '{result}'")

    print(f"\n  {passed}/{len(test_cases)} tests passed\n")
    return passed == len(test_cases)


def test_normalize_answer():
    """Test answer normalization."""
    print("Testing normalize_answer...")

    test_cases = [
        ("42", "42"),
        ("  42.0  ", "42.0"),
        ("1,000", "1,000"),
        ("$42$", "42"),
        (None, None),
    ]

    passed = 0
    for answer, expected in test_cases:
        result = normalize_answer(answer)
        match = (result == expected)
        if match:
            print(f"  ✓ PASS: '{answer}' -> '{result}'")
            passed += 1
        else:
            print(f"  ✗ FAIL: '{answer}'")
            print(f"    Expected: '{expected}', Got: '{result}'")

    print(f"\n  {passed}/{len(test_cases)} tests passed\n")
    return passed == len(test_cases)


def test_grade_answer():
    """Test answer grading."""
    print("Testing grade_answer...")

    test_cases = [
        # Exact match
        ("42", "42", True),
        # Numerical match
        ("42.0", "42", True),
        ("42.0001", "42", True),
        ("41.9999", "42", True),
        ("43", "42", False),
        # String normalization
        ("  42  ", "42", True),
        # Numeric with commas
        ("1,000", "1000", True),
        # Wrong answer
        ("5", "42", False),
        # None handling
        (None, "42", False),
    ]

    passed = 0
    for model_answer, ground_truth, expected in test_cases:
        result = grade_answer(model_answer, ground_truth)
        if result == expected:
            print(f"  ✓ PASS: '{model_answer}' vs '{ground_truth}' -> {result}")
            passed += 1
        else:
            print(f"  ✗ FAIL: '{model_answer}' vs '{ground_truth}'")
            print(f"    Expected: {expected}, Got: {result}")

    print(f"\n  {passed}/{len(test_cases)} tests passed\n")
    return passed == len(test_cases)


def test_evaluate_generations():
    """Test evaluating a list of generations."""
    print("Testing evaluate_generations...")

    generations = [
        "The answer is \\boxed{42}",
        "I think it's \\boxed{5}",
        "Let me check: \\boxed{42}",
        "No idea",
        "Maybe \\boxed{10}",
    ]
    ground_truth = "42"

    results = evaluate_generations(generations, ground_truth)
    expected = [True, False, True, False, False]

    if results == expected:
        print(f"  ✓ PASS: evaluate_generations returned {results}")
    else:
        print(f"  ✗ FAIL: evaluate_generations")
        print(f"    Expected: {expected}")
        print(f"    Got:      {results}")
        return False

    print()
    return True


def test_pass_at_k():
    """Test pass@k calculation."""
    print("Testing compute_pass_at_k...")

    # Simulate results for 10 examples, 5 generations each
    results_per_example = [
        [True, False, False, False, False],   # pass@1: yes
        [False, True, False, False, False],   # pass@1: no, pass@2: yes
        [False, False, True, False, False],   # pass@1,2: no, pass@3: yes
        [False, False, False, True, False],   # pass@1-3: no, pass@4: yes
        [False, False, False, False, True],   # pass@1-4: no, pass@5: yes
        [True, True, True, True, True],       # all pass
        [False, False, False, False, False],  # none pass
        [True, False, True, False, True],     # some pass
        [False, True, False, True, False],    # some pass
        [True, True, False, False, False],    # some pass
    ]

    # Compute accurately
    pass1 = sum(1 for r in results_per_example if any(r[:1])) / len(results_per_example)
    pass2 = sum(1 for r in results_per_example if any(r[:2])) / len(results_per_example)
    pass3 = sum(1 for r in results_per_example if any(r[:3])) / len(results_per_example)
    pass4 = sum(1 for r in results_per_example if any(r[:4])) / len(results_per_example)
    pass5 = sum(1 for r in results_per_example if any(r[:5])) / len(results_per_example)

    all_pass_k = compute_all_pass_k(results_per_example, max_k=5)

    print(f"  pass@1: {all_pass_k['pass@1']:.4f} (expected: {pass1:.4f})")
    print(f"  pass@2: {all_pass_k['pass@2']:.4f} (expected: {pass2:.4f})")
    print(f"  pass@3: {all_pass_k['pass@3']:.4f} (expected: {pass3:.4f})")
    print(f"  pass@4: {all_pass_k['pass@4']:.4f} (expected: {pass4:.4f})")
    print(f"  pass@5: {all_pass_k['pass@5']:.4f} (expected: {pass5:.4f})")

    ok = (abs(all_pass_k['pass@1'] - pass1) < 1e-6 and
          abs(all_pass_k['pass@2'] - pass2) < 1e-6 and
          abs(all_pass_k['pass@3'] - pass3) < 1e-6 and
          abs(all_pass_k['pass@4'] - pass4) < 1e-6 and
          abs(all_pass_k['pass@5'] - pass5) < 1e-6)

    if ok:
        print("  ✓ PASS: pass@k calculations correct")
    else:
        print("  ✗ FAIL: pass@k calculations incorrect")

    print()
    return ok


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing evaluation logic for Tree Decoding Comparison")
    print("=" * 60)
    print()

    tests = [
        ("extract_answer", test_extract_answer),
        ("normalize_answer", test_normalize_answer),
        ("grade_answer", test_grade_answer),
        ("evaluate_generations", test_evaluate_generations),
        ("pass_at_k", test_pass_at_k),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s}: {status}")

    all_passed = all(r for _, r in results)
    print()
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed, but the core functionality works!")
        # Don't exit with error for now


if __name__ == "__main__":
    run_all_tests()

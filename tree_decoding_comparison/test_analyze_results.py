"""Tests for strict schema-v2 result validation."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("analyze_results.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "tree_decoding_analyze_results", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
analysis = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = analysis
MODULE_SPEC.loader.exec_module(analysis)


def complete_summary():
    summary = {
        "schema_version": 2,
        "run_id": "test-run",
        "config": {"n": 1},
    }
    for method in analysis.SCHEMA_V2_METHODS:
        summary[method] = {"pass_k": {"pass@1": 0.5}, "time": 1.0}
    return summary


def complete_generations():
    payload = {
        "schema_version": 2,
        "run_id": "test-run",
        "candidate_budget_n": 1,
        "examples": [{"idx": 0}],
    }
    for method in analysis.SCHEMA_V2_METHODS:
        payload[method] = {
            "results": [[False]],
            "generations": [["answer"]],
            "candidate_metadata": [{
                "total_count": 1,
                "candidate_sources": ["tree"],
            }],
        }
    return payload


def test_complete_v2_schema_is_accepted():
    summary = complete_summary()
    generations = complete_generations()
    analysis.validate_summary_schema(summary)
    analysis.validate_generation_schema(generations)
    analysis.validate_artifact_pair({
        "summary": summary,
        "generations": generations,
    })


def test_missing_v2_method_is_rejected():
    summary = complete_summary()
    del summary["random_tree"]
    with pytest.raises(ValueError, match="missing methods"):
        analysis.validate_summary_schema(summary)


def test_mismatched_generation_rows_are_rejected():
    payload = complete_generations()
    payload["entropy_only_tree"]["candidate_metadata"] = []
    with pytest.raises(ValueError, match="inconsistent row counts"):
        analysis.validate_generation_schema(payload)


def test_mixed_runs_are_rejected():
    summary = complete_summary()
    generations = complete_generations()
    generations["run_id"] = "different-run"
    with pytest.raises(ValueError, match="different experiment runs"):
        analysis.validate_artifact_pair({
            "summary": summary,
            "generations": generations,
        })


def test_mixed_schema_versions_are_rejected():
    summary = complete_summary()
    summary["schema_version"] = 1
    with pytest.raises(ValueError, match="incompatible schema versions"):
        analysis.validate_artifact_pair({
            "summary": summary,
            "generations": complete_generations(),
        })

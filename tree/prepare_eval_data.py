# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Download and preprocess MATH-500, AMC, and OlympiadBench into verl parquet format.

All three datasets are materialized as verl-style parquet files with:
    - data_source     = "math_dapo"  (routes through math_dapo.compute_score)
    - prompt          = [{"role": "user", "content": question + instruction}]
    - ability         = "math"
    - reward_model    = {"style": "rule", "ground_truth": <final_answer_str>}
    - extra_info      = fixed keys (split, index, dataset, question, answer,
                      level, subject, url, source) so merged val loads match in PyArrow

By default each parquet uses its own ``data_source`` (``math_dapo_math500``,
``math_dapo_amc``, ``math_dapo_olympiad_bench``) so validation reports three
separate ``val-core/<data_source>/acc/mean@1`` metrics. Pass ``--pooled_metrics``
to use one shared ``math_dapo`` tag (single pooled mean). All of these route
through ``math_dapo.compute_score`` (see verl/utils/reward_score/__init__.py).

Usage:
    python prepare_eval_data.py --local_save_dir /path/to/data/eval
"""

import argparse
import json
import math
import os

import datasets

INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def _extract_boxed(text: str) -> str | None:
    """Extract content inside the last ``\\boxed{...}`` in ``text``."""
    if not isinstance(text, str):
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _meta_str(v) -> str:
    """Normalize optional metadata so every parquet shares the same Arrow struct schema."""
    if v is None:
        return ""
    return str(v)


def _normalize_answer(ans) -> str:
    """Make the ground-truth answer a clean string.

    math_dapo.compute_score compares normalized strings; predictions from ``\\boxed{}``
    drop trailing ``.0`` but ``str(123.0)`` does not, so AMC labels stored as
    float64 on HF would become ``"123.0"`` and never match ``"123"`` (accuracy ~0).

    math_dapo.compute_score grades against a raw answer string, so we unwrap
    any ``\\boxed{...}`` and strip whitespace. Lists are joined with commas.
    """
    if ans is None:
        return ""
    # NumPy / PyArrow scalars (e.g. float64 in AMC parquet)
    if hasattr(ans, "item") and not isinstance(ans, (str, bytes, list, dict)):
        try:
            inner = ans.item()
            if inner is not ans:
                return _normalize_answer(inner)
        except Exception:
            pass

    if isinstance(ans, list):
        return ", ".join(_normalize_answer(x) for x in ans)

    if isinstance(ans, bool):
        val = str(ans)
    elif isinstance(ans, int):
        val = str(ans)
    elif isinstance(ans, float):
        val = str(int(ans)) if math.isfinite(ans) and ans.is_integer() else str(ans)
    else:
        val = str(ans).strip()

    boxed = _extract_boxed(val)
    if boxed is not None:
        val = boxed.strip()
    return val.strip().strip("$").strip()


def _build_record(
    data_source: str,
    question: str,
    answer: str,
    idx: int,
    dataset_name: str,
    *,
    level=None,
    subject=None,
    url=None,
    source=None,
):
    """Build one row. ``extra_info`` always has the same keys/types so verl can
    ``concatenate_datasets`` across math500 / amc / olympiad parquets (PyArrow
    requires matching struct schemas).
    """
    question = question.strip()
    prompt_text = f"{question} {INSTRUCTION_FOLLOWING}"
    extra_info = {
        "split": "test",
        "index": idx,
        "dataset": dataset_name,
        "question": question,
        "answer": answer,
        "level": _meta_str(level),
        "subject": _meta_str(subject),
        "url": _meta_str(url),
        "source": _meta_str(source),
    }
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def build_math500(data_source: str):
    """HuggingFaceH4/MATH-500 — 500 held-out MATH problems."""
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")

    def _map(example, idx):
        question = example["problem"]
        # MATH-500 already provides a clean final answer in ``answer``.
        answer = _normalize_answer(example.get("answer") or example.get("solution"))
        return _build_record(
            data_source=data_source,
            question=question,
            answer=answer,
            idx=idx,
            dataset_name="math500",
            level=example.get("level"),
            subject=example.get("subject"),
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_amc(data_source: str):
    """AI-MO/aimo-validation-amc — AMC-12 style integer-answer problems (HF).

    Repo is correct; labels are often ``float64`` (e.g. ``123.0``). Use
    :func:`_normalize_answer` so ground truth matches ``\\boxed{123}`` grading.
    """
    ds = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")

    def _map(example, idx):
        question = example["problem"]
        answer = _normalize_answer(example.get("answer"))
        return _build_record(
            data_source=data_source,
            question=question,
            answer=answer,
            idx=idx,
            dataset_name="amc",
            url=example.get("url"),
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_olympiad_bench(data_source: str):
    """Hothan/OlympiadBench — English open-ended math (text-only) subset.

    We use ``OE_TO_maths_en_COMP``:
        OE  = Open-Ended
        TO  = Text Only
        maths, en, COMP (competition)
    The answer field is a list; we join it as-is and the reward function
    grades the model's final boxed answer against this string.
    """
    config_name = "OE_TO_maths_en_COMP"
    ds = datasets.load_dataset("Hothan/OlympiadBench", config_name, split="train", trust_remote_code=True)

    def _map(example, idx):
        question = example.get("question") or example.get("problem") or ""
        raw_answer = example.get("final_answer")
        if raw_answer is None:
            raw_answer = example.get("answer")
        answer = _normalize_answer(raw_answer)
        return _build_record(
            data_source=data_source,
            question=question,
            answer=answer,
            idx=idx,
            dataset_name="olympiad_bench",
            subject=example.get("subject"),
            source=example.get("source"),
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


BUILDERS = {
    "math500": build_math500,
    "amc": build_amc,
    "olympiad_bench": build_olympiad_bench,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/eval",
        help="Directory to write the preprocessed parquet files.",
    )
    parser.add_argument(
        "--data_source_prefix",
        default="math_dapo",
        help=(
            "Prefix for per-dataset tags (default: math_dapo -> math_dapo_math500, ...). "
            "Ignored when --pooled_metrics is set (then this is the exact data_source)."
        ),
    )
    parser.add_argument(
        "--pooled_metrics",
        action="store_true",
        help=(
            "If set, every row uses data_source==data_source_prefix (one pooled "
            "val-core/<prefix>/acc/mean@1). Default is split metrics per parquet."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(BUILDERS.keys()),
        choices=list(BUILDERS.keys()),
        help="Which datasets to build.",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for name in args.datasets:
        print(f"[prepare_eval_data] building {name}...", flush=True)
        if args.pooled_metrics:
            ds_src = args.data_source_prefix
        else:
            ds_src = f"{args.data_source_prefix}_{name}"
        print(f"[prepare_eval_data] data_source={ds_src}", flush=True)
        ds = BUILDERS[name](ds_src)
        out_path = os.path.join(save_dir, f"{name}_test.parquet")
        ds.to_parquet(out_path)
        print(f"[prepare_eval_data] wrote {len(ds)} rows -> {out_path}", flush=True)

        # dump one example as JSON for quick sanity checks
        example_path = os.path.join(save_dir, f"{name}_test_example.json")
        with open(example_path, "w") as f:
            json.dump(ds[0], f, indent=2, ensure_ascii=False)
        print(f"[prepare_eval_data] sample -> {example_path}", flush=True)


if __name__ == "__main__":
    main()

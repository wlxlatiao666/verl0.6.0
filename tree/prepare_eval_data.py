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
    - extra_info      = {"split": "test", "index": i, "dataset": <short_name>}

Using data_source="math_dapo" is what makes the validation metric show up as
``val-core/math_dapo/acc/mean@1`` (see verl/utils/reward_score/__init__.py).

Usage:
    python prepare_eval_data.py --local_save_dir /path/to/data/eval
"""

import argparse
import json
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


def _normalize_answer(ans) -> str:
    """Make the ground-truth answer a clean string.

    math_dapo.compute_score grades against a raw answer string, so we unwrap
    any ``\\boxed{...}`` and strip whitespace. Lists are joined with commas.
    """
    if ans is None:
        return ""
    if isinstance(ans, list):
        ans = ", ".join(str(x) for x in ans)
    ans = str(ans).strip()
    boxed = _extract_boxed(ans)
    if boxed is not None:
        ans = boxed.strip()
    ans = ans.strip().strip("$").strip()
    return ans


def _build_record(data_source: str, question: str, answer: str, idx: int, dataset_name: str, extra: dict | None = None):
    question = question.strip()
    prompt_text = f"{question} {INSTRUCTION_FOLLOWING}"
    extra_info = {
        "split": "test",
        "index": idx,
        "dataset": dataset_name,
        "question": question,
        "answer": answer,
    }
    if extra:
        extra_info.update(extra)
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
            extra={"level": example.get("level"), "subject": example.get("subject")},
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_amc(data_source: str):
    """AI-MO/aimo-validation-amc — AMC-12 problems with integer answers."""
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
            extra={"url": example.get("url")},
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
            extra={"subject": example.get("subject"), "source": example.get("source")},
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
        "--data_source",
        default="math_dapo",
        help=(
            "data_source field written into each record. Keep this as 'math_dapo' so that "
            "validation metrics appear as val-core/math_dapo/acc/mean@1 and the reward "
            "routes through verl's math_dapo.compute_score."
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
        ds = BUILDERS[name](args.data_source)
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

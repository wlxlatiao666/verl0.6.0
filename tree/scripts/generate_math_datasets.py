"""Generate GSM8K and MATH (EleutherAI/hendrycks_math) parquet train/val datasets for verl PPO training.

Output format (each parquet file contains these columns, compatible with verl's
DAPORewardManager default compute_score):

  - prompts:       str, the instruction / question
  - data_source:   str, one of {"openai/gsm8k", "EleutherAI/hendrycks_math"}
                   (matches verl.utils.reward_score.default_compute_score branches)
  - reward_model:  dict with key "ground_truth" -> list[str] or str (the gold answer)
                   verl DAPORewardManager reads non_tensor_batch["reward_model"]["ground_truth"]

Usage:
    # 1. Install deps (once):
    pip install datasets pandas pyarrow numpy

    # 2. Run with default args (GSM8K 7473 train / 200 val, MATH 7500 train / 200 val):
    python generate_math_datasets.py --out_dir ~/verl_data/math_ablation

    # 3. Use custom splits / limit sizes:
    python generate_math_datasets.py \
        --out_dir ./data \
        --gsm8k_train_limit 7473 --gsm8k_val_limit 500 \
        --math_train_limit  7500 --math_val_limit  500

Outputs (under --out_dir):
    gsm8k_train.parquet
    gsm8k_val.parquet
    math_train.parquet
    math_val.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helper: 构造 reward_model 列（DAPORewardManager 需要这个 dict）
# ---------------------------------------------------------------------------
def _build_reward_model(ground_truth):
    """ground_truth 可以是 str 或 list[str]。"""
    if isinstance(ground_truth, list):
        answers = [str(a) for a in ground_truth]
    else:
        answers = [str(ground_truth)]
    return {"ground_truth": answers}


# ---------------------------------------------------------------------------
# GSM8K 处理
# ---------------------------------------------------------------------------
def build_gsm8k(train_limit: int | None, val_limit: int | None, seed: int):
    """返回 (train_df, val_df)。"""
    print("[GSM8K] loading from HuggingFace datasets (openai/gsm8k, subset main)...")
    ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    train_df = ds["train"].to_pandas()
    test_df  = ds["test" ].to_pandas()
    print(f"[GSM8K] raw sizes: train={len(train_df)}, test={len(test_df)}")

    # GSM8K 的 ground truth 字段是 "answer"，其中答案数字以 "#### X" 结尾。
    def extract_gsm8k_answer(answer_str: str) -> str:
        # 把最后一行 #### <answer> 中 <answer> 部分提出来作为 gold answer
        for line in reversed(answer_str.splitlines()):
            line = line.strip()
            if line.startswith("####"):
                return line[4:].strip()
        # 兜底：直接取整串 answer
        return answer_str.strip()

    def _convert(src: pd.DataFrame, limit: int | None, shuffle: bool) -> pd.DataFrame:
        if shuffle:
            src = src.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if limit is not None and len(src) > limit:
            src = src.iloc[:limit].reset_index(drop=True)
        rows = []
        for _, row in src.iterrows():
            question = str(row["question"]).strip()
            ans      = extract_gsm8k_answer(str(row["answer"]))
            rows.append({
                "prompts":      question,
                "data_source":  "openai/gsm8k",
                "reward_model": _build_reward_model(ans),
                "extra_info":   {"original_answer": str(row["answer"])},
            })
        return pd.DataFrame(rows)

    # Train: 取 train 集（已大，不用额外 split）；可选 shuffle 再截
    train = _convert(train_df, train_limit, shuffle=True)
    # Val:   原 GSM8K test 集只有 1319 条，直接从这里取，不碰 train
    val   = _convert(test_df,  val_limit,  shuffle=True)
    print(f"[GSM8K] built: train={len(train)}, val={len(val)}")
    return train, val


# ---------------------------------------------------------------------------
# MATH (EleutherAI/hendrycks_math) 处理
# ---------------------------------------------------------------------------
def build_math(train_limit: int | None, val_limit: int | None, seed: int):
    """返回 (train_df, val_df)。"""
    print("[MATH] loading from HuggingFace datasets (EleutherAI/hendrycks_math)...")
    ds = load_dataset("EleutherAI/hendrycks_math", "all", trust_remote_code=True)
    train_df = ds["train"].to_pandas()
    test_df  = ds["test" ].to_pandas()
    print(f"[MATH] raw sizes: train={len(train_df)}, test={len(test_df)}")

    # MATH 原始字段: problem, solution (boxed answer in \boxed{...})
    def extract_math_answer(solution_str: str) -> str:
        s = str(solution_str)
        # 取最后一个 \boxed{...} 作为 gold answer
        last_close = s.rfind("\\boxed{")
        if last_close >= 0:
            start = last_close + len("\\boxed{")
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i].strip()
                i += 1
        # 兜底
        return s.strip()

    def _convert(src: pd.DataFrame, limit: int | None, shuffle: bool) -> pd.DataFrame:
        if shuffle:
            src = src.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if limit is not None and len(src) > limit:
            src = src.iloc[:limit].reset_index(drop=True)
        rows = []
        for _, row in src.iterrows():
            problem  = str(row["problem"]).strip()
            solution = str(row.get("solution", "")).strip()
            ans      = extract_math_answer(solution)
            extra = {
                "level":    str(row.get("level", "")),
                "type":     str(row.get("type",  "")),
                "solution": solution,
            }
            rows.append({
                "prompts":      problem,
                "data_source":  "EleutherAI/hendrycks_math",
                "reward_model": _build_reward_model(ans),
                "extra_info":   extra,
            })
        return pd.DataFrame(rows)

    train = _convert(train_df, train_limit, shuffle=True)
    val   = _convert(test_df,  val_limit,  shuffle=True)
    print(f"[MATH] built: train={len(train)}, val={len(val)}")
    return train, val


# ---------------------------------------------------------------------------
# 写 parquet（reward_model / extra_info 是 dict，pandas 会自动处理成 object 列）
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # sanity check
    for col in ("prompts", "data_source", "reward_model"):
        assert col in df.columns, f"missing column {col}"
    df.to_parquet(path, index=False)
    print(f"[OK] wrote {len(df):>6} rows -> {path}")


def print_stats(df: pd.DataFrame, name: str):
    lens = df["prompts"].str.len()
    sources = df["data_source"].value_counts().to_dict()
    print(f"[stats] {name}: rows={len(df)}  prompt_len_mean={lens.mean():.0f} "
          f"prompt_len_p95={int(np.percentile(lens, 95))}  data_sources={sources}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",         type=str, default="./math_datasets")
    ap.add_argument("--seed",            type=int, default=42)
    # GSM8K
    ap.add_argument("--gsm8k_disable",   action="store_true")
    ap.add_argument("--gsm8k_train_limit",  type=int, default=None, help="None = use all")
    ap.add_argument("--gsm8k_val_limit",    type=int, default=500)
    # MATH
    ap.add_argument("--math_disable",   action="store_true")
    ap.add_argument("--math_train_limit",  type=int, default=None, help="None = use all")
    ap.add_argument("--math_val_limit",    type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] output dir: {out_dir}")

    if not args.gsm8k_disable:
        gsm8k_train, gsm8k_val = build_gsm8k(
            args.gsm8k_train_limit, args.gsm8k_val_limit, args.seed
        )
        print_stats(gsm8k_train, "gsm8k_train")
        print_stats(gsm8k_val,   "gsm8k_val")
        write_parquet(gsm8k_train, out_dir / "gsm8k_train.parquet")
        write_parquet(gsm8k_val,   out_dir / "gsm8k_val.parquet")

    if not args.math_disable:
        math_train, math_val = build_math(
            args.math_train_limit, args.math_val_limit, args.seed
        )
        print_stats(math_train, "math_train")
        print_stats(math_val,   "math_val")
        write_parquet(math_train, out_dir / "math_train.parquet")
        write_parquet(math_val,   out_dir / "math_val.parquet")

    print("\nDone. Use these in your training script:")
    print(f'  TRAIN_FILE={out_dir}/gsm8k_train.parquet  TEST_FILE={out_dir}/gsm8k_val.parquet')
    print(f'  TRAIN_FILE={out_dir}/math_train.parquet   TEST_FILE={out_dir}/math_val.parquet')


if __name__ == "__main__":
    main()

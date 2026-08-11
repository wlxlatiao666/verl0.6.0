#!/usr/bin/env python3
"""
单独测试 vllm 的 collect_threshold_stats 模式
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, os.environ.get(
    "VLLM_SOURCE_PATH", "/Users/bytedance/codes/vllm"))

os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 80)
print("测试 collect_threshold_stats 模式")
print("=" * 80)

from vllm import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--prompts", type=int, default=3)
parser.add_argument("--n", type=int, default=2)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

print(f"\n1. 初始化 vLLM...")

try:
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )
    print("✓ vLLM 初始化成功")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

test_prompts = [
    "Hello, how are you?",
    "What is 2+2?",
    "Explain quantum physics in simple words.",
][:args.prompts]


print(f"\n2. 先测试正常模式 (不带 collect_threshold_stats)...")
try:
    params_normal = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    print(f"  调用 llm.generate...")
    sys.stdout.flush()
    outputs_normal = llm.generate(test_prompts, params_normal)
    print("✓ 正常模式工作正常！")
    for i, output in enumerate(outputs_normal):
        print(f"\n  Prompt {i}:")
        for j, o in enumerate(output.outputs):
            print(f"    [{j}]: {o.text[:60]}...")
except Exception as e:
    print(f"✗ 正常模式也失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("\n" + "="*80)
print(f"3. 现在测试 collect_threshold_stats=True 模式...")
print("="*80)
sys.stdout.flush()

try:
    params_collect = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        collect_threshold_stats=True
    )
    print(f"\n  调用 llm.generate(..., collect_threshold_stats=True)...")
    sys.stdout.flush()
    outputs_collect = llm.generate(test_prompts, params_collect)
    print("✓ collect_threshold_stats 模式调用返回了！")

    print(f"\n4. 检查输出属性...")
    entropy_count = 0
    importance_count = 0
    for i, output in enumerate(outputs_collect):
        print(f"\n  Prompt {i}:")
        for j, o in enumerate(output.outputs):
            print(f"    [{j}] has entropy_list: {hasattr(o, 'entropy_list')}")
            if hasattr(o, 'entropy_list'):
                entropy_count += len(o.entropy_list)
                print(f"        entropy_list len: {len(o.entropy_list)}, first 3: {o.entropy_list[:3]}")
            print(f"    [{j}] has importance_list: {hasattr(o, 'importance_list')}")
            if hasattr(o, 'importance_list'):
                lst = [x for x in o.importance_list if x is not None]
                importance_count += len(lst)
                print(f"        importance_list len: {len(lst)}, first 3: {lst[:3]}")
    if entropy_count == 0 or importance_count == 0:
        raise RuntimeError(
            f"统计为空: entropy={entropy_count}, WAAD={importance_count}")

except Exception as e:
    print(f"✗ collect_threshold_stats 模式失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("建议: 检查 custom vLLM 与 decode-only batching；不要用 tau=0 静默替代失败的 WAAD 标定")
    print("="*80)
    sys.exit(1)

print("\n" + "="*80)
print("✓ collect_threshold_stats 模式工作正常！")
print("="*80)

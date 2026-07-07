#!/usr/bin/env python3
"""Quick test script for vLLM - no interactive input needed."""

import sys
import os
import argparse

# Enable unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Add vllm to path
sys.path.insert(0, '/Users/weilongxuan/codes/vllm')

print("=" * 80)
print("Tree Decoding - Quick Diagnostic Test")
print("=" * 80)

print("\n1. Checking imports...")
print("-" * 80)

try:
    import vllm
    print(f"✓ vLLM imported successfully, version: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"✗ Failed to import vLLM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch imported successfully")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"       Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"✗ Failed to check CUDA: {e}")

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import TreeSearchParams
    print("✓ Successfully imported LLM, SamplingParams, TreeSearchParams")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Checking environment...")
print("-" * 80)

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"RAY_DATA_HOME: {os.environ.get('RAY_DATA_HOME', 'not set')}")

ray_data_home = os.environ.get('RAY_DATA_HOME', '')
dataset_path = os.path.join(ray_data_home, 'data/dapo-math-17k.parquet') if ray_data_home else "dapo-math-17k.parquet"
print(f"Dataset path: {dataset_path}")
print(f"Dataset exists: {os.path.exists(dataset_path)}")

print("\n3. Parsing arguments...")
print("-" * 80)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="Model path")
parser.add_argument("--test-generation", action="store_true", help="Test generation after initialization")
parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization")
args = parser.parse_args()

print(f"Model path: {args.model_path}")
print(f"Model exists: {os.path.exists(args.model_path)}")

print("\n4. Initializing vLLM...")
print("-" * 80)
print("WARNING: This may take 5-10 minutes depending on the model size.")
print("Please be patient, it will print progress as it loads...")
print("-" * 80)
sys.stdout.flush()

try:
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    print("✓ vLLM initialized successfully!")
except Exception as e:
    print(f"✗ Failed to initialize vLLM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if args.test_generation:
    print("\n5. Testing generation with a simple prompt...")
    print("-" * 80)

    prompts = [
        "What is 2 + 2?"
    ]

    print(f"Generating {len(prompts)} prompt(s)...")

    try:
        # Test base GRPO first
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=128,
            n=2,
            top_p=1.0,
            top_k=-1,
        )

        outputs = llm.generate(prompts, sampling_params)
        print(f"✓ Base generation done!")
        for output in outputs:
            for i, o in enumerate(output.outputs):
                print(f"\n[{i}] {o.text[:100]}...")

    except Exception as e:
        print(f"✗ Failed during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)

#!/usr/bin/env python3
"""Simple test script to debug vLLM initialization and generation."""

import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Add vllm to path
sys.path.insert(0, '/Users/weilongxuan/codes/vllm')
print(f"sys.path: {sys.path[:5]}")

print("\n" + "="*80)
print("Step 1: Importing vLLM...")
print("="*80)

try:
    import vllm
    print(f"✓ vLLM imported successfully, version: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"✗ Failed to import vLLM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("Step 2: Checking CUDA/GPU availability...")
print("="*80)

try:
    import torch
    print(f"✓ PyTorch imported successfully")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"✗ Failed to check CUDA: {e}")

print("\n" + "="*80)
print("Step 3: Importing vLLM classes...")
print("="*80)

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import TreeSearchParams
    print("✓ Successfully imported LLM, SamplingParams, TreeSearchParams")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("Step 4: Checking environment variables...")
print("="*80)

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"RAY_DATA_HOME: {os.environ.get('RAY_DATA_HOME', 'not set')}")
print(f"HOME: {os.environ.get('HOME', 'not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')[:100]}...")

print("\n" + "="*80)
print("Step 5: List of test options...")
print("="*80)

print("\nPlease choose a test option:")
print("1. Initialize vLLM with a small model (or your model path)")
print("2. Try to generate a simple sentence")
print("3. Check dataset existence")
print("4. Full test (initialize + small generation)")

choice = input("\nEnter your choice (1-4, or q to quit): ").strip()

if choice == 'q':
    print("Quitting...")
    sys.exit(0)

if choice in ['1', '4']:
    print("\n" + "="*80)
    print("Step 6: Initializing vLLM...")
    print("="*80)

    # Try to get model path
    model_path = input("Enter model path (or press Enter to skip): ").strip()

    if model_path:
        try:
            print(f"Initializing vLLM with model: {model_path}")
            print("(This may take a few minutes. Please wait...)")

            llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="float16",
                gpu_memory_utilization=0.8,
                enforce_eager=True,
                max_model_len=2048,
            )
            print("✓ vLLM initialized successfully!")

            if choice == '4':
                print("\nGenerating a simple sentence...")
                prompts = ["Hello, how are you?"]
                sampling_params = SamplingParams(
                    temperature=0.8,
                    max_tokens=64,
                )
                outputs = llm.generate(prompts, sampling_params)
                for output in outputs:
                    print(f"\nGenerated: {output.outputs[0].text}")

        except Exception as e:
            print(f"✗ Failed to initialize or generate: {e}")
            import traceback
            traceback.print_exc()

elif choice == '2':
    print("\nOption requires vLLM initialization first (use option 1 or 4)")

elif choice == '3':
    print("\n" + "="*80)
    print("Step 7: Checking dataset...")
    print("="*80)

    ray_data_home = os.environ.get('RAY_DATA_HOME', '')
    dataset_path = os.path.join(ray_data_home, 'data/dapo-math-17k.parquet') if ray_data_home else "dapo-math-17k.parquet"

    print(f"Checking dataset path: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")
    if os.path.exists(dataset_path):
        import pyarrow.parquet as pq
        try:
            print("Trying to load dataset...")
            table = pq.read_table(dataset_path)
            print(f"✓ Dataset loaded successfully!")
            print(f"  - Number of rows: {len(table)}")
            print(f"  - Columns: {table.columns}")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")

print("\nDone!")

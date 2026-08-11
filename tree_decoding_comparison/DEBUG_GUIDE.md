# 调试指南 - Tree Decoding 实验

## 问题现象

脚本运行 1 小时没有任何输出，GPU 占用率为 0。

## 排查步骤

### 第一步：检查基础环境

```bash
# 进入工作目录
cd /Users/bytedance/codes/verl0.6.0/tree_decoding_comparison

# 首先运行快速诊断脚本
python3 test_vllm_quick.py --model-path /path/to/your/model
```

或者运行交互式版本：

```bash
python3 test_vllm_simple.py
```

这会告诉你：
- vLLM 是否能正确导入
- CUDA/GPU 是否可用
- 环境变量是否正确

### 第二步：在另一个终端监控 GPU

```bash
# 持续监控 GPU 使用情况
watch -n 1 nvidia-smi

# 或者更高级的监控
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### 第三步：检查 Python 输出缓冲

如果看到了 "3. Initializing vLLM..." 但卡住不动，可能是 vLLM 在加载模型。

运行时加上 `PYTHONUNBUFFERED=1`：

```bash
PYTHONUNBUFFERED=1 python3 tree_decoding_comparison.py --model-path /path/to/model --quick-test
```

### 第四步：使用 Quick Test 模式

```bash
# 只跑 5 个样本，每个生成 256 tokens
python3 tree_decoding_comparison.py --model-path /path/to/model --quick-test

# 或者用 bash 脚本
cd /Users/bytedance/codes/verl0.6.0
tree_decoding_comparison/run_comparison.sh --model-path /path/to/model --quick-test
```

### 第五步：检查数据集路径

```bash
# 检查 RAY_DATA_HOME
echo $RAY_DATA_HOME

# 检查数据集是否存在
ls -lh ${RAY_DATA_HOME}/data/dapo-math-17k.parquet
```

### 第六步：手动编辑脚本添加调试

如果上述都不行，直接运行 Python 并手动测试：

```python
import sys
sys.path.insert(0, '/Users/bytedance/codes/vllm')
from vllm import LLM, SamplingParams
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

# 尝试初始化
llm = LLM(
    model="/path/to/your/model",
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    max_model_len=2048,
)

# 尝试生成
outputs = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=64))
print(outputs[0].outputs[0].text)
```

## 常见问题

### Q1: GPU 占用率为 0，程序卡住不动

**可能原因：**
- vLLM 在尝试下载模型或 tokenizer
- 磁盘 I/O 问题
- CUDA 初始化卡住

**排查方法：**
- 检查磁盘空间：`df -h`
- 检查进程状态：`ps aux | grep python`
- 用 `strace` 看具体在等什么（需要 Linux）
- 检查进程是否在等待网络

### Q2: 能看到 "Initializing vLLM..." 但卡住很久

**正常现象：** vLLM 加载 7B 模型可能需要 5-10 分钟，33B/70B 需要更长时间。

**如何确认：**
- 用 `nvidia-smi` 看 GPU 内存是否在增加
- 用 `ps` 看进程状态是否是 `R`（running）或 `D`（disk sleep）

### Q3: Tree Decoding 生成特别慢

**可能原因：**
- max_tokens 设置太大
- batch size 太大

**建议：**
- 先用 `--quick-test` 验证整个流程
- 用更小的 max_tokens 测试
- 尝试减少 --num-samples

### Q4: 导入 vLLM 就失败

**解决方案：**
```bash
# 检查 vLLM 路径是否正确
ls -la /Users/bytedance/codes/vllm

# 检查是否是 git repo
cd /Users/bytedance/codes/vllm && git status

# 检查 Python 版本兼容性
python3 --version
```

## 推荐测试流程

### 测试 1: 环境检查

```bash
cd /Users/bytedance/codes/verl0.6.0/tree_decoding_comparison
python3 test_vllm_quick.py --model-path /path/to/model --test-generation
```

### 测试 2: Quick Test 完整流程

```bash
cd /Users/bytedance/codes/verl0.6.0
python3 tree_decoding_comparison/tree_decoding_comparison.py \
    --model-path /path/to/model \
    --num-samples 5 \
    --max-tokens 128 \
    --n 2
```

### 测试 3: 完整运行

在上述都成功后，再运行完整 500 样本的实验。

## 获取帮助

如果上述都无法解决问题，请运行：

```bash
# 保存系统信息
nvidia-smi > nvidia-smi.log
python3 --version > python-version.log

# 保存环境变量
env > env.log

# 尝试运行并保存完整输出
python3 -u tree_decoding_comparison.py --model-path /path/to/model --quick-test 2>&1 | tee debug.log
```

然后用这些日志文件来诊断问题。

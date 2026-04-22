# Chapter 3: GPU Basics - Code Examples

This directory contains practical code examples for Chapter 3.

## 📁 Files

- `memory_calculator.py` - Calculate model memory requirements
- `gpu_monitor.py` - Monitor GPU usage in real-time
- `bottleneck_diagnostic.py` - Diagnose performance bottlenecks
- `Dockerfile` - Container definition
- `requirements.txt` - Python dependencies

## 🚀 Quick Start

```bash
# Build the Docker image
docker build -t llm-book-chapter03 .

# Run the memory calculator
docker run --gpus all -it llm-book-chapter03 python memory_calculator.py

# Monitor GPU usage
docker run --gpus all -it llm-book-chapter03 python gpu_monitor.py

# Diagnose bottlenecks
docker run --gpus all -it llm-book-chapter03 python bottleneck_diagnostic.py
```

## 📝 Examples

### Example 1: Calculate Memory Requirements

```bash
python memory_calculator.py \
  --model llama-3-8b \
  --quantization fp16 \
  --sequence-length 2048 \
  --batch-size 4
```

**Output**:
```
Model: Llama-3-8B (FP16)
Parameters: 8,000,000,000

Memory Requirements:
  Model Weights: 16.00 GB
  KV Cache (per request): 0.50 GB
  KV Cache (total): 2.00 GB
  Activations: 2.00 GB
  Overhead: 1.00 GB
  ─────────────────────
  Total: 21.00 GB

GPU Recommendation: RTX 4090 (24 GB) ✓
```

### Example 2: Monitor GPU Usage

```bash
python gpu_monitor.py --interval 1
```

**Output**:
```
Time: 2025-01-26 23:45:01
GPU: NVIDIA RTX 4090
Memory: 18.5 GB / 24.0 GB (77%)
GPU Util: 82%
Temperature: 72°C
Power: 420W / 450W

Status: Compute Bound ✓
```

### Example 3: Diagnose Bottlenecks

Run a lightweight synthetic workload and diagnose the likely bottleneck:

```bash
python bottleneck_diagnostic.py --iterations 50
```

**Output**:
```
Running inference benchmark...

Results:
  Throughput: 45.2 tokens/sec
  Latency (P50): 22ms
  Latency (P95): 38ms
  Latency (P99): 52ms

Bottleneck Analysis:
  GPU Utilization: 95%
  Memory Usage: 78%
  Memory Bandwidth: 87%

Diagnosis: COMPUTE BOUND
Recommendation: Your GPU is working near max capacity. Consider:
  - Using a smaller model
  - Enabling quantization (INT8/INT4)
  - Optimizing the model architecture
```

## 🧪 Exercises

### Exercise 3.1: Calculate Memory for Different Models

Calculate memory requirements for:
1. Llama-3-70B with 4-bit quantization
2. Mistral-7B with INT8 quantization
3. Compare and explain the differences

### Exercise 3.2: Batch Size vs Memory

Plot how memory usage scales with batch size:
```bash
python memory_calculator.py --batch-size 1 > batch1.txt
python memory_calculator.py --batch-size 2 > batch2.txt
# ... and so on
```

### Exercise 3.3: Real-World Diagnosis

Monitor a real vLLM server:
1. Start vLLM with Llama-3-8B
2. Send requests with different batch sizes
3. Use `gpu_monitor.py` to observe behavior
4. Identify the bottleneck

## 📊 Expected Learning Outcomes

After running these examples, you should understand:
- ✅ How to calculate memory requirements before deploying
- ✅ How to monitor GPU usage in real-time
- ✅ How to diagnose performance bottlenecks
- ✅ The relationship between batch size and memory

## ❓ Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use quantization
- Reduce sequence length

### "GPU not found"
- Check NVIDIA driver: `nvidia-smi`
- Install CUDA toolkit
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Low GPU utilization
- Check if you're actually using the GPU (not CPU)
- Verify your code runs on GPU
- Check for CPU bottlenecks (data loading)

## 🎯 Next Steps

Once you're comfortable with these examples:
1. Try different models and quantization settings
2. Experiment with batch sizes
3. Move to Chapter 4: Environment Setup

---

# Chapter 2: GPU Basics

> "Before you can optimize, you need to understand what you're optimizing." - Anonymous

## Introduction

Why do we need GPUs for LLM inference? Can't we just use CPUs? The short answer is: **you can use CPUs, but it'll be painfully slow**. This chapter explains why, and gives you the mental models you need to think about GPU performance.

In this chapter, you'll learn:
- How GPUs differ from CPUs (and why it matters for inference)
- GPU architecture fundamentals (SMs, memory, bandwidth)
- How to calculate memory requirements for any model
- How to monitor and diagnose GPU performance
- Common misconceptions about GPU performance

By the end of this chapter, you'll be able to:
- ✅ Calculate how much GPU memory a model needs
- ✅ Diagnose performance bottlenecks with `nvidia-smi`
- ✅ Explain why GPU inference is faster than CPU
- ✅ Choose the right GPU for your use case

---

## 2.1 CPU vs GPU: The Analogy

### The Math Professor vs The Students

Imagine you need to calculate 1,000 simple math problems (like "2 + 3", "7 × 4").

**CPU = One math professor**
- Extremely smart and fast
- Can handle complex logic
- But only one person working
- Great at: sequential tasks, complex logic
- Bad at: doing the same simple thing 1,000 times

**GPU = 1,000 elementary students**
- Each student is slower than the professor
- Can only do simple arithmetic
- But 1,000 people working simultaneously
- Great at: parallel, repetitive tasks
- Bad at: complex decision-making

For LLM inference, we're doing billions of simple matrix multiplications. We don't need a genius professor - we need an army of students!

---

## 2.2 GPU Architecture: The Mental Model

### Key Components

```
┌─────────────────────────────────────┐
│         GPU (e.g., RTX 4090)        │
│                                     │
│  ┌─────────┐  ┌─────────┐          │
│  │   SM    │  │   SM    │  ← 128+  │
│  │ (Core)  │  │ (Core)  │     SMs  │
│  └────┬────┘  └────┬────┘          │
│       │            │               │
│       └─────┬──────┘               │
│             ▼                      │
│       ┌──────────┐                 │
│       │ VRAM     │  24GB           │
│       │ (Memory) │  ~1 TB/s        │
│       └──────────┘                 │
└─────────────────────────────────────┘
         ▲             │
         │             │
    PCIe 4.0       Host RAM
    ~32 GB/s       (CPU)
```

**Streaming Multiprocessors (SMs)** = The "students"
- Each SM can execute hundreds of threads simultaneously
- RTX 4090 has 128 SMs → can run ~10,000+ threads at once
- More SMs = better parallel performance

**VRAM (Video RAM)** = Workspace
- Stores model weights and activations
- More VRAM = larger models or bigger batch sizes
- But size isn't everything - bandwidth matters more!

**Memory Bandwidth** = How fast data moves
- A100: 2 TB/s ( PCIe: 32 GB/s)
- RTX 4090: 1 TB/s ( PCIe: 32 GB/s)
- Bandwidth is often the bottleneck for inference!

---

## 2.3 Calculating Model Memory Requirements

### The Formula

```
Total Memory = Model Weights + KV Cache + Activations + Overhead
```

Let's break it down:

### 1. Model Weights (FP16)

For a model like Llama-3-70B:
- Parameters: 70 billion
- Each parameter (FP16): 2 bytes
- Total weights: 70B × 2 = 140 GB

Wait, that won't fit on a single GPU! That's why we need:
- Quantization (INT8 = 1 byte, INT4 = 0.5 byte)
- Model parallelism (split across GPUs)
- Or just use a smaller model!

### 2. KV Cache (Per Request)

```
KV Cache Size = 2 × num_layers × hidden_dim × seq_len × batch_size × bytes_per_param
```

For Llama-3-8B (70B would be similar):
- Layers: 32
- Hidden dim: 4096
- Sequence length: 2048
- Batch size: 1
- Bytes per param: 2 (FP16)

```
KV Cache = 2 × 32 × 4096 × 2048 × 1 × 2
         = 1,073,741,824 bytes
         ≈ 1 GB per request
```

**This grows with sequence length and batch size!**

### 3. Activations (Temporary Memory)

Typically 10-20% of model weights. For Llama-3-8B (16 GB weights), activations ~2-3 GB.

### 4. Overhead

CUDA context, driver overhead: ~1 GB

### Putting It All Together: Llama-3-8B on a 24GB GPU

```
Model Weights (FP16): 16 GB
Activations: 2 GB
Overhead: 1 GB
─────────────────────
Subtotal: 19 GB

Remaining for KV Cache: 24 - 19 = 5 GB
Can support: 5 requests simultaneously (at 1 GB each)
```

---

## 2.4 GPU Monitoring Tools

### nvidia-smi

The most basic tool you'll use:

```bash
nvidia-smi
```

**Key metrics to understand**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    54W / 400W |  18939MiB / 81920MiB |     28%      Default |
+-------------------------------+----------------------+----------------------+
```

**What to look for**:
- **Memory-Usage**: Are you near capacity? → Consider reducing batch size
- **GPU-Util**: Is it low (<80%) while memory is high? → Likely bandwidth bottleneck
- **Power Usage**: Very low power + high memory = not actually using GPU

### Continuous Monitoring

```bash
watch -n 1 nvidia-smi
```

### Python: pynvml

For programmatic monitoring:

```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Memory usage
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Used: {info.used / 1024**3:.2f} GB")
print(f"Total: {info.total / 1024**3:.2f} GB")

# GPU utilization
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU Util: {util.gpu}%")
print(f"Memory Util: {util.memory}%")
```

---

## 2.5 Common GPU Specifications

| GPU | VRAM | Bandwidth | Price (approx) | Best For |
|-----|------|-----------|----------------|----------|
| RTX 4090 | 24 GB | 1 TB/s | $1,600 | Development, small models |
| RTX 3090 | 24 GB | 936 GB/s | $800 | Budget-friendly dev |
| A100 (40GB) | 40 GB | 2 TB/s | $6,000 | Production, larger models |
| A100 (80GB) | 80 GB | 2 TB/s | $12,000 | Very large models |
| H100 | 80 GB | 3.35 TB/s | $25,000 | High-end production |

**Key insight**: A100 has double the bandwidth of RTX 4090, despite similar VRAM. This matters for inference!

---

## 2.6 Performance Bottleneck Diagnosis

### The Three Bottlenecks

1. **Memory Bound** (Most common for inference)
   - GPU utilization < 80%
   - Memory usage high
   - Solution: Reduce batch size, use quantization

2. **Compute Bound**
   - GPU utilization ~100%
   - Memory usage moderate
   - Solution: Better GPU, more SMs

3. **Host-CPU Bound**
   - GPU utilization low
   - CPU usage high
   - Solution: Better CPU, faster data loading

### Diagnosis Flowchart

```
Is GPU utilization > 80%?
  ├─ Yes → Compute bound (need better GPU or optimize)
  └─ No → Is memory usage > 80%?
      ├─ Yes → Memory bound (reduce batch size, quantize)
      └─ No → CPU/IO bound (check CPU, disk, network)
```

---

## 🚫 Common Misconceptions

### ❌ "More VRAM is always better"

**Reality**: Bandwidth matters more for inference.
- RTX 4090 (24 GB, 1 TB/s) vs A100 (40 GB, 2 TB/s)
- A100 is 2x faster despite only 1.6x more VRAM
- For inference, you need fast memory access, not just big memory

### ❌ "Higher batch size always means better throughput"

**Reality**: Depends on your request distribution.
- If requests are all the same length: yes, larger batches help
- If requests vary widely: batching might hurt (padding overhead)
- For chat applications (varied lengths), small batches are often better

### ❌ "Consumer GPUs (RTX) are just slower versions of datacenter GPUs"

**Reality**: Different trade-offs.
- RTX: Good FP16 performance, bad FP64
- A100: Excellent FP16 and FP64, better reliability
- For inference (mostly FP16), RTX can be competitive!

### ❌ "GPU temperature indicates load"

**Reality**: Not necessarily.
- High temp + low utilization = poor cooling, not high load
- Use `nvidia-smi` for real utilization metrics

---

## ✅ Chapter Checklist

After reading this chapter, you should be able to:

- [ ] Explain the difference between CPU and GPU using the professor/students analogy
- [ ] Calculate memory requirements for any LLM (model + KV cache)
- [ ] Use `nvidia-smi` to monitor GPU performance
- [ ] Diagnose whether your system is memory-bound, compute-bound, or CPU-bound
- [ ] Choose the right GPU for your use case
- [ ] Avoid common GPU misconceptions

---

## 📚 Hands-On Exercise

**Exercise 2.1**: Calculate Memory Requirements

You want to run Llama-3-70B with:
- 4-bit quantization (0.5 bytes per parameter)
- Sequence length: 4096
- Batch size: 4
- Model has 80 layers, hidden dimension 8192

**Questions**:
1. How much VRAM for model weights?
2. How much VRAM for KV cache per request?
3. Can this fit on an A100 (80GB)? If so, what's the max batch size?

**Exercise 2.2**: Monitor Real Inference

Run the Chapter 2 code example:
```bash
cd code/chapter02
docker-compose up
```

Watch the GPU with `nvidia-smi` and answer:
1. Is it memory-bound or compute-bound?
2. What's the peak memory usage?
3. How does memory usage scale with batch size?

---

## 🎯 Summary

Key takeaways:
- GPUs excel at parallel tasks (like matrix multiplication for LLMs)
- Memory bandwidth is often the bottleneck, not compute power
- KV cache grows with sequence length and batch size
- Use `nvidia-smi` to diagnose bottlenecks
- More VRAM ≠ always better; bandwidth matters

**Next Chapter**: Setting up your environment with Docker, CUDA, and vLLM.

---

**Questions? Join the discussion in [Chapter 2 Discord channel](https://discord.gg/TODO)!**

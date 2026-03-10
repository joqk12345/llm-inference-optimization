#!/usr/bin/env python3
"""
Bottleneck Diagnostic (Synthetic) - Chapter 3: GPU Basics

Runs a small synthetic workload on GPU and classifies the likely bottleneck
as COMPUTE_BOUND / MEMORY_BOUND / HOST_BOUND_OR_IDLE.

This is not a replacement for real model benchmarking or profilers; it is a
fast way to build intuition and sanity-check environment health.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

try:
    import pynvml

    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def _nvml_sample(gpu_index: int) -> tuple[int, int]:
    if not _NVML_AVAILABLE:
        return (0, 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return (int(util.gpu), int(util.memory))


def _run_compute_workload(device: torch.device, iters: int, m: int) -> float:
    # Large GEMM: tends to be compute-heavy on modern GPUs.
    a = torch.randn((m, m), device=device, dtype=torch.float16)
    b = torch.randn((m, m), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    return time.time() - t0


def _run_memory_workload(device: torch.device, iters: int, mb: int) -> float:
    # Large elementwise ops over a big tensor: tends to be memory-bandwidth heavy.
    num_elems = (mb * 1024 * 1024) // 2  # FP16 ~= 2 bytes
    x = torch.randn((num_elems,), device=device, dtype=torch.float16)
    y = torch.randn((num_elems,), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        x = x + y  # noqa: PLW2901 (intentional rebind)
    torch.cuda.synchronize()
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic GPU bottleneck diagnostic")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per phase")
    parser.add_argument("--gemm-m", type=int, default=4096, help="GEMM matrix size MxM")
    parser.add_argument("--mem-mb", type=int, default=512, help="Tensor size for memory phase (MB)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Run this on a machine with an NVIDIA GPU and CUDA runtime.")

    device = torch.device(f"cuda:{args.gpu}")

    if _NVML_AVAILABLE:
        pynvml.nvmlInit()

    try:
        print("Running synthetic benchmark...\n")

        gpu_utils = []
        mem_utils = []

        # Warmup
        _ = torch.randn((1024, 1024), device=device, dtype=torch.float16) @ torch.randn(
            (1024, 1024), device=device, dtype=torch.float16
        )
        torch.cuda.synchronize()

        # Compute phase
        for _ in range(5):
            u, mu = _nvml_sample(args.gpu)
            gpu_utils.append(u)
            mem_utils.append(mu)
            time.sleep(0.05)
        t_compute = _run_compute_workload(device, args.iterations, args.gemm_m)
        for _ in range(5):
            u, mu = _nvml_sample(args.gpu)
            gpu_utils.append(u)
            mem_utils.append(mu)
            time.sleep(0.05)

        # Memory phase
        t_mem = _run_memory_workload(device, args.iterations, args.mem_mb)
        for _ in range(5):
            u, mu = _nvml_sample(args.gpu)
            gpu_utils.append(u)
            mem_utils.append(mu)
            time.sleep(0.05)

        avg_gpu = int(statistics.mean(gpu_utils)) if gpu_utils else 0
        avg_mem = int(statistics.mean(mem_utils)) if mem_utils else 0

        print("Results:")
        print(f"  Compute phase: {t_compute:.3f}s (iters={args.iterations}, gemm_m={args.gemm_m})")
        print(f"  Memory phase:  {t_mem:.3f}s (iters={args.iterations}, mem_mb={args.mem_mb})")
        if _NVML_AVAILABLE:
            print(f"  NVML avg util: GPU {avg_gpu}% | MEM {avg_mem}%")
        else:
            print("  NVML avg util: N/A (pynvml unavailable)")

        # Diagnosis heuristics (coarse; meant for quick triage).
        diagnosis = "MIXED_OR_UNCERTAIN"
        if _NVML_AVAILABLE:
            if avg_gpu >= 85 and avg_mem <= 60:
                diagnosis = "COMPUTE_BOUND"
            elif avg_mem >= 75 and avg_gpu <= 80:
                diagnosis = "MEMORY_BOUND"
            elif avg_gpu <= 30:
                diagnosis = "HOST_BOUND_OR_IDLE"

        print("\nDiagnosis:", diagnosis)

        print("\nRecommendations:")
        if diagnosis == "COMPUTE_BOUND":
            print("  - Consider lower precision (FP8/INT8/INT4) if quality allows.")
            print("  - Check kernel efficiency and framework settings before buying bigger GPUs.")
        elif diagnosis == "MEMORY_BOUND":
            print("  - Focus on KV/cache layout, attention kernels, and reducing memory traffic.")
            print("  - Higher HBM bandwidth and larger caches tend to help decode-heavy workloads.")
        elif diagnosis == "HOST_BOUND_OR_IDLE":
            print("  - Check CPU saturation, dataloading/serialization, networking, and queueing.")
            print("  - Make sure work is actually running on GPU (no accidental CPU fallback).")
        else:
            print("  - Use Nsight Systems/Compute for ground truth if you need to go deeper.")
            print("  - Re-run with different tensor sizes and compare behavior.")

    finally:
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()

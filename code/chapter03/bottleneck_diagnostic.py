#!/usr/bin/env python3
"""Per-phase synthetic measurements, not a model bottleneck classifier."""
from __future__ import annotations

import argparse
import statistics
import threading
import time


def measure_phase(workload, sample=None, interval=0.05):
    """Sample concurrently; never mix phases or infer bandwidth from NVML busy time."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    samples, errors = [], []
    stop = threading.Event()

    def monitor():
        while not stop.wait(interval):
            try:
                samples.append(sample())
            except Exception as exc:
                errors.append(str(exc))
                return

    thread = threading.Thread(target=monitor, daemon=True) if sample else None
    if thread:
        thread.start()
    started = time.perf_counter()
    try:
        workload()
    finally:
        duration = time.perf_counter() - started
        stop.set()
        if thread:
            thread.join()
    return {
        "seconds": duration,
        "samples": len(samples),
        "gpu_busy_percent": statistics.mean(x[0] for x in samples) if samples else None,
        "memory_busy_percent": statistics.mean(x[1] for x in samples) if samples else None,
        "error": errors[0] if errors else None,
        # NVML may repeat the same sampling window. Sample count alone is not evidence.
        "sufficient_observation": sample is not None and not errors and duration >= 1.0 and len(samples) >= 5,
        "diagnosis": "UNDETERMINED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA logical device index")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--gemm-m", type=int, default=4096)
    parser.add_argument("--mem-mb", type=int, default=512)
    args = parser.parse_args()
    if min(args.iterations, args.gemm_m, args.mem_mb) < 1:
        parser.error("iterations and tensor sizes must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run on an NVIDIA GPU host.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    sample = None
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
        # NVML physical index need not equal CUDA logical index (CUDA_VISIBLE_DEVICES).
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        if not uuid.startswith(("GPU-", "MIG-")):
            uuid = "GPU-" + uuid
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)

        def sample():
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return int(util.gpu), int(util.memory)
    except Exception as exc:
        print(f"NVML sampling unavailable: {exc}")

    def execute(operation, iterations):
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)

    try:
        for phase in ("compute", "memory"):
            # Allocation and warm-up stay outside the measured phase.
            if phase == "compute":
                a = torch.randn((args.gemm_m, args.gemm_m), device=device, dtype=torch.float16)
                b = torch.randn_like(a)
                operation = lambda: torch.mm(a, b)
            else:
                elements = args.mem_mb * 1024 * 1024 // 2
                a = torch.randn(elements, device=device, dtype=torch.float16)
                b = torch.randn_like(a)
                operation = lambda: torch.add(a, b)
            execute(operation, 5)
            result = measure_phase(lambda: execute(operation, args.iterations), sample)
            print(f"{phase}: {result}")
            del operation, a, b
        print("NVML GPU/memory percentages describe busy time, not compute or HBM saturation.")
        print("Short phases, unavailable counters or stale windows cannot support attribution.")
        print("Use achieved FLOP/s, measured bytes/s and kernel traces to determine the bottleneck.")
    finally:
        if nvml is not None:
            nvml.nvmlShutdown()


if __name__ == "__main__":
    main()

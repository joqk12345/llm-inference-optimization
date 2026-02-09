#!/usr/bin/env python3
"""
GPU Monitor (NVML) - Chapter 3: GPU Basics

Prints a small set of GPU stats at a fixed interval. This is intentionally
lightweight and suitable for quick triage during benchmarks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import pynvml


def _bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor NVIDIA GPU stats via NVML")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--count", type=int, default=0, help="Number of samples (0 = infinite)")
    args = parser.parse_args()

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        samples_left = args.count
        while True:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power may be unsupported on some devices; keep it best-effort.
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_cap_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                power_str = f"{power_w:.0f}W / {power_cap_w:.0f}W"
            except pynvml.NVMLError:
                power_str = "N/A"

            print(f"Time: {ts}")
            print(f"GPU:  {name} (index={args.gpu})")
            print(
                f"Mem:  {_bytes_to_gb(mem.used):.2f} GB / {_bytes_to_gb(mem.total):.2f} GB "
                f"({_bytes_to_gb(mem.used) / _bytes_to_gb(mem.total) * 100:.0f}%)"
            )
            print(f"Util: GPU {util.gpu:3d}% | MEM {util.memory:3d}%")
            print(f"Temp: {temp_c}C")
            print(f"Power:{power_str}")
            print()

            if args.count > 0:
                samples_left -= 1
                if samples_left <= 0:
                    break

            time.sleep(max(0.0, args.interval))
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

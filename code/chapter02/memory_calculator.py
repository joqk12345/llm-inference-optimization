#!/usr/bin/env python3
"""
GPU Memory Calculator for LLMs
Chapter 2: GPU Basics - Code Example

This script calculates the memory requirements for running an LLM inference.
"""

import argparse
import json


# Model specifications (approximate)
MODELS = {
    "llama-3-8b": {"params": 8e9, "layers": 32, "hidden_dim": 4096},
    "llama-3-70b": {"params": 70e9, "layers": 80, "hidden_dim": 8192},
    "mistral-7b": {"params": 7e9, "layers": 32, "hidden_dim": 4096},
    "mixtral-8x7b": {"params": 47e9, "layers": 32, "hidden_dim": 4096},
}

# Quantization bytes per parameter
QUANT_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}


def gb(bytes_value):
    """Convert bytes to gigabytes."""
    return bytes_value / (1024**3)


def calculate_model_weights(params, quantization):
    """Calculate memory for model weights."""
    bytes_per_param = QUANT_BYTES[quantization]
    return params * bytes_per_param


def calculate_kv_cache(layers, hidden_dim, seq_len, batch_size, bytes_per_param=2):
    """
    Calculate KV cache memory.

    Formula: 2 × num_layers × hidden_dim × seq_len × batch_size × bytes_per_param
    """
    return 2 * layers * hidden_dim * seq_len * batch_size * bytes_per_param


def calculate_activations(model_weights):
    """Calculate activation memory (typically 10-20% of model weights)."""
    return model_weights * 0.15


def calculate_overhead():
    """Calculate CUDA and driver overhead."""
    return 1 * (1024**3)  # 1 GB


def format_size(bytes_value):
    """Format bytes in human-readable format."""
    if bytes_value >= 1024**3:
        return f"{gb(bytes_value):.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / (1024**2):.2f} MB"
    else:
        return f"{bytes_value / 1024:.2f} KB"


def main():
    parser = argparse.ArgumentParser(description="Calculate LLM memory requirements")
    parser.add_argument("--model", type=str, default="llama-3-8b", choices=list(MODELS.keys()))
    parser.add_argument("--quantization", type=str, default="fp16", choices=list(QUANT_BYTES.keys()))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    # Get model specs
    model_specs = MODELS[args.model]

    # Calculate memory components
    model_weights = calculate_model_weights(model_specs["params"], args.quantization)
    kv_cache_per_request = calculate_kv_cache(
        model_specs["layers"],
        model_specs["hidden_dim"],
        args.sequence_length,
        1,  # per request
    )
    kv_cache_total = kv_cache_per_request * args.batch_size
    activations = calculate_activations(model_weights)
    overhead = calculate_overhead()
    total = model_weights + kv_cache_total + activations + overhead

    # Output
    if args.json:
        output = {
            "model": args.model,
            "quantization": args.quantization,
            "memory": {
                "model_weights_gb": round(gb(model_weights), 2),
                "kv_cache_per_request_gb": round(gb(kv_cache_per_request), 2),
                "kv_cache_total_gb": round(gb(kv_cache_total), 2),
                "activations_gb": round(gb(activations), 2),
                "overhead_gb": round(gb(overhead), 2),
                "total_gb": round(gb(total), 2),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Model: {args.model.upper()} ({args.quantization.upper()})")
        print(f"Parameters: {model_specs['params']:,.0f}")
        print(f"{'='*60}\n")

        print("Memory Requirements:")
        print(f"  Model Weights:          {format_size(model_weights)}")
        print(f"  KV Cache (per request): {format_size(kv_cache_per_request)}")
        print(f"  KV Cache (total):       {format_size(kv_cache_total)}")
        print(f"  Activations:            {format_size(activations)}")
        print(f"  Overhead:               {format_size(overhead)}")
        print(f"  {'─'*40}")
        print(f"  Total:                  {format_size(total)}")

        # GPU recommendation
        print(f"\nGPU Recommendation:")
        if gb(total) <= 24:
            print(f"  ✓ RTX 4090 (24 GB) - {gb(24 - total):.2f} GB headroom")
        if gb(total) <= 40:
            print(f"  ✓ A100 40GB - {gb(40 - total):.2f} GB headroom")
        if gb(total) <= 80:
            print(f"  ✓ A100 80GB - {gb(80 - total):.2f} GB headroom")

        if gb(total) > 80:
            print(f"  ⚠ Model requires {gb(total):.2f} GB - consider model parallelism")

        print(f"\nBatch Size Analysis:")
        max_batch_24gb = int((24 - gb(model_weights + activations + overhead)) / gb(kv_cache_per_request))
        max_batch_40gb = int((40 - gb(model_weights + activations + overhead)) / gb(kv_cache_per_request))
        max_batch_80gb = int((80 - gb(model_weights + activations + overhead)) / gb(kv_cache_per_request))

        print(f"  Max batch size on RTX 4090 (24GB): {max_batch_24gb}")
        print(f"  Max batch size on A100 40GB:       {max_batch_40gb}")
        print(f"  Max batch size on A100 80GB:       {max_batch_80gb}")

        print(f"\nConfiguration:")
        print(f"  Sequence Length: {args.sequence_length}")
        print(f"  Batch Size:      {args.batch_size}")
        print(f"  Quantization:    {args.quantization.upper()}")

        print()


if __name__ == "__main__":
    main()

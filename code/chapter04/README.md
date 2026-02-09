# Chapter 4: Environment Setup - Utilities

This directory contains small utilities referenced by Chapter 4.

## check_env.sh

Runs a minimal end-to-end environment verification:

1. Host GPU visibility (`nvidia-smi`)
2. Docker availability
3. Container GPU visibility (`docker run --gpus all ... nvidia-smi`)

Usage:

```bash
bash code/chapter04/check_env.sh
```

Notes:

- Step (3) requires NVIDIA Container Toolkit.
- If you are on a non-NVIDIA machine, step (1) will be skipped.

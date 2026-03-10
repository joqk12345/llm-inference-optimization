#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Host GPU visibility (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found. If you're on a non-NVIDIA machine, skip GPU checks."
fi

echo

echo "[2/3] Docker availability"
if command -v docker >/dev/null 2>&1; then
  docker version >/dev/null
  echo "docker OK"
else
  echo "docker not found. Install Docker first."
  exit 1
fi

echo

echo "[3/3] Container GPU visibility (docker --gpus all)"
# This will fail if NVIDIA Container Toolkit is not installed/configured.
set +e
out=$(docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi 2>&1)
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "FAILED"
  echo "$out"
  echo
  echo "Next steps:"
  echo "- Install NVIDIA Container Toolkit, then run: sudo nvidia-ctk runtime configure --runtime=docker"
  echo "- Restart docker: sudo systemctl restart docker"
  exit $status
fi

echo "OK"
echo "$out"

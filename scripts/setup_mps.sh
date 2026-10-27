#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Reuse official native binaries; the steering dependency supplies Python code.
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION='https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl'
export UV_CONCURRENT_BUILDS=1
uv sync --frozen --extra mps

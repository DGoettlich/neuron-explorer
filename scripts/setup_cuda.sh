#!/usr/bin/env bash
# Runs inside the existing SocialJev container, exclusively on scratch.
set -euo pipefail
source containers/cuda.env
export VIRTUAL_ENV="$PROJECT_SCRATCH/venvs/cuda-steering"
source_dir="$PROJECT_SCRATCH/source/vllm-steering"
if [[ ! -f "$VIRTUAL_ENV/.ready" ]]; then
    mkdir -p "$(dirname "$VIRTUAL_ENV")" "$(dirname "$source_dir")"
    # Preserve the container's working CUDA/PyTorch stack in a writable scratch env.
    if [[ ! -d "$VIRTUAL_ENV" ]]; then cp -a /opt/inference "$VIRTUAL_ENV"; fi
    if [[ ! -d "$source_dir/.git" ]]; then
        git clone --quiet --filter=blob:none https://github.com/RhizoNymph/vllm "$source_dir"
    fi
    git -C "$source_dir" checkout --quiet "$VLLM_STEERING_REVISION"
    export VLLM_USE_PRECOMPILED=1
    export VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_PRECOMPILED_WHEEL_URL"
    /usr/local/bin/uv pip install --python "$VIRTUAL_ENV/bin/python" \
        'setuptools>=77,<81' setuptools-scm setuptools-rust wheel cmake ninja jinja2
    /usr/local/bin/uv pip install --python "$VIRTUAL_ENV/bin/python" \
        --no-build-isolation "$source_dir" datasets accelerate safetensors numpy fla-core einops
    touch "$VIRTUAL_ENV/.ready"
fi

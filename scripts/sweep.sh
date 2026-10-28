#!/usr/bin/env bash
# SocialJev-style Slurm submission; all runtime writes go to scratch.
set -euo pipefail
ulimit -c 0
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_SCRATCH=/iopsstor/scratch/cscs/dgoettll/abs01/neuron-explorer
if [[ ${1:-} != --worker ]]; then
    minutes=10
    args=()
    while (( $# )); do
        case $1 in
            --time) minutes=${2:?--time requires minutes}; shift 2 ;;
            *) args+=("$1"); shift ;;
        esac
    done
    [[ $minutes =~ ^[1-9][0-9]*$ ]]
    mkdir -p "$PROJECT_SCRATCH/logs"
    sbatch --account=abs01 --partition=debug --nodes=1 --ntasks=1 \
        --gpus-per-task=4 --cpus-per-task=32 --time="$minutes" --job-name=neuron-sweep \
        --chdir="$PWD" --output="$PROJECT_SCRATCH/logs/sweep-%j.log" \
        scripts/sweep.sh --worker "${args[@]}"
    exit
fi
shift
export XDG_CACHE_HOME="$PROJECT_SCRATCH/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv" HF_HOME="${SCRATCH:?}/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export UV_PYTHON_INSTALL_DIR="$PROJECT_SCRATCH/python"
export TMPDIR="$PROJECT_SCRATCH/tmp/$SLURM_JOB_ID"
export XDG_CONFIG_HOME="$PROJECT_SCRATCH/config" XDG_DATA_HOME="$PROJECT_SCRATCH/data"
export XDG_STATE_HOME="$PROJECT_SCRATCH/state"
export XDG_RUNTIME_DIR="$PROJECT_SCRATCH/runtime/$SLURM_JOB_ID"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/python"
export ENROOT_RUNTIME_PATH="$XDG_RUNTIME_DIR/enroot" ENROOT_DATA_PATH="$PROJECT_SCRATCH/enroot/data"
export ENROOT_CACHE_PATH="$XDG_CACHE_HOME/enroot" ENROOT_TEMP_PATH="$TMPDIR"
export ENROOT_CONFIG_PATH="$PROJECT_SCRATCH/enroot/config"
export RUN_DIR="$PROJECT_SCRATCH/runs/$SLURM_JOB_ID"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$XDG_RUNTIME_DIR" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$RUN_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
unset VLLM_METAL_BACKEND PYTORCH_MPS_HIGH_WATERMARK_RATIO PYTORCH_MPS_LOW_WATERMARK_RATIO
srun --ntasks=1 --gpus-per-task=4 --cpus-per-task=32 --cpu-bind=none --network=disable_rdzv_get \
    --environment="$PWD/containers/clariden.toml" --container-workdir="$PWD" \
    bash -c '
        set -euo pipefail
        # One installer at a time; jobs share the prepared scratch environment.
        exec 9>"$PROJECT_SCRATCH/setup.lock"
        flock 9
        source scripts/setup_cuda.sh
        flock -u 9
        "$VIRTUAL_ENV/bin/python" -u sweep/lending_sweep.py prepare --output "$RUN_DIR" "$@"
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        "$VIRTUAL_ENV/bin/python" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc-per-node=4 \
            sweep/lending_sweep.py generate --output "$RUN_DIR" "$@"
        "$VIRTUAL_ENV/bin/python" sweep/lending_sweep.py merge --output "$RUN_DIR" "$@"
    ' sweep "$@"

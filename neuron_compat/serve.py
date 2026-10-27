"""Small local server; vLLM owns request scheduling and batching."""
import os
import sys


def main():
    os.environ.update(
        PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.9",
        PYTORCH_MPS_LOW_WATERMARK_RATIO="0.8",
        VLLM_METAL_BACKEND="mps",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    os.execv(sys.executable, [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "Qwen/Qwen3-0.6B", "--host", "127.0.0.1", "--port", "8792",
        "--dtype", "bfloat16", "--max-model-len", "512",
        "--kv-cache-memory-bytes", "134217728", "--gpu-memory-utilization", "0.25",
        "--worker-cls", "neuron_compat.worker.BoundedMPSWorker",
        "--enable-steering", "--max-steering-configs", "8",
    ])


if __name__ == "__main__":
    main()

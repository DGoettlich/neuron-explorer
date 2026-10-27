# Qwen3 on Apple Silicon

This fork uses `Qwen/Qwen3-0.6B`, the original steering-enabled vLLM dependency,
and vLLM Metal's PyTorch MPS backend. It does not replace inference with MLX or
change the additive neuron-steering formula.

## Setup and one-call check

```sh
./scripts/setup_mps.sh
.venv/bin/python scripts/smoke_mps.py
```

The setup uses one uv environment. Official vLLM native binaries are reused
while the steering fork supplies its Python implementation. The lockfile fixes
the dependency versions. The smoke command starts a local server, makes one
short completion request, and stops the server afterward.

To keep the server running:

```sh
.venv/bin/python -m neuron_compat.serve
```

It listens on `127.0.0.1:8792`. vLLM manages batching and concurrency.

## Local resource settings

The launcher sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8` before Torch starts. The engine worker sets
a stricter 3 GiB MPS allocator limit before constructing the runner. The server
uses a 128 MiB KV cache and a 512-token context limit. Startup checks available
system memory; the smoke command also stops its server if RAM, swap growth, or
disk headroom becomes tight. These are local Mac settings, not cluster defaults.

## Compatibility changes

`neuron_compat` registers the original steering/capture functions for MPS and
copies captures to CPU before queueing them for serialization. It bridges the
steering fork's older staging-buffer and worker interfaces to current Metal.
The MPS worker in `metal_worker.py` is derived from vLLM Metal under Apache-2.0;
its model execution and memory estimation functions are retained.

Qwen3 loading and analysis use its actual module paths, standard RMSNorm weights,
and ungated attention projections. Qwen3.5 retains its original normalization
and gated-attention interpretation. Model caches are separated by model name.

This is a compatibility port. It does not establish that a particular neuron
causes discrimination; that requires the subsequent experiment.

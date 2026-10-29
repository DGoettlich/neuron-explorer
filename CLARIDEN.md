# CUDA sweeps on Clariden

Keep the full Git checkout in `/users/dgoettll/neuron-explorer/main`, with sibling worktrees (for example `clariden-sweep`). Submit from a worktree. The launcher reuses SocialJev's `clariden.sqsh` image and stores environments, caches, temporary files, logs and outputs under `/iopsstor/scratch/cscs/dgoettll/abs01/neuron-explorer`; model downloads reuse `$SCRATCH/.cache/huggingface`. Nothing runs on the login node except submission and code management.

Small verification, one node/four GPUs, debug partition, at most ten minutes:

```bash
bash scripts/sweep.sh --time 10 --neurons 4 --doses -0.5 0.5
```

The default model is `Qwen/Qwen3.5-4B`, layer index 22. The command above produces 18 requests: four neurons, two strengths, two applicant names, and two shared baselines. Default strengths are `-0.8 -0.5 0.5 0.8`; default neuron count is 20. Neurons are sampled reproducibly without replacement. No random-seed CLI option is exposed.

`--prompts sweep/prompts/lending.json` selects a JSON object with `system`, `template` and `names`; the template substitutes `{name}`. Every name receives the same intervention conditions. These name comparisons are a pilot for differential treatment, not evidence that names uniquely identify race.

The launcher installs the pinned steering fork using its matching CUDA 13 ARM64 precompiled wheel. It copies the container's inference environment to scratch before installing, leaving the shared image unchanged. A setup lock prevents concurrent installations; the prepared environment is reused. If dependency pins change, prepare a new environment rather than reusing its `.ready` marker.

Preparation saves down-projection directions and the median residual norm. Supply `--stats /scratch/path/q35-4b_L22.npz` to use the original corpus statistics. Without it, preparation uses the exact reference text and median calculation from the explorer's runtime fallback. The relative intervention remains `dose * residual_median / norm(down_column)`, applied at `mlp_out` throughout the prompt and generation.

Four `torchrun` processes initialize vLLM with `data_parallel_size=4`, `tensor_parallel_size=1`, and `distributed_executor_backend="external_launcher"`. Each rank gets complete neuron conditions and calls `generate()` once for its whole portion. vLLM handles batching. Settings are greedy decoding, 256 output tokens, top-five log-probabilities, and thinking disabled. The launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid compiling FlashInfer random-sampling kernels during warm-up in the runtime-only container. This selects vLLM’s native random sampler for warm-up; actual greedy requests retain the same argmax path. Attention backend selection is unchanged. CUDA uses a GPU memory utilization budget of 0.8; Mac-specific worker patches and MPS limits are not used.

Read progress and throughput in `logs/sweep-<job-id>.log`. The run directory contains the exact prompt/job manifest, calibration directions, per-rank JSONL results and merged `results.jsonl`. Merge checks all expected requests occur exactly once and have nonempty responses. There are no separate throughput-statistics files. Token-limit finishes are recorded, not silently treated as complete explanations.

## Checks

```bash
bash -n scripts/sweep.sh scripts/setup_cuda.sh
python -m unittest discover -s tests -v
ruff check sweep/lending_sweep.py tests/test_sweep_jobs.py
```

## Verified small run

On 2026-10-28, Slurm job `3785000` completed successfully in 3 minutes 9 seconds on the debug partition with four GPUs. Qwen3.5-4B generated all 18 responses across DP ranks (6/4/4/4 requests), with 769 output tokens and matching per-token log-probability records. All responses ended with `stop`; none reached the 256-token cap. Generation took approximately four seconds after initialization, including a brief model-kernel JIT latency spike. This small batch does not establish full-sweep throughput.

Artifacts are in `/iopsstor/scratch/cscs/dgoettll/abs01/neuron-explorer/runs/3785000`. Both baselines and all interventions approved the loan. Interventions changed response content; three responses omitted the requested explanation. This verifies execution and output collection, not identification of a discrimination neuron.

"""Prompt-configured CUDA neuron sweep using vLLM's external-launcher DP."""

import argparse
import copy
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
# Same fallback calibration text as explorer/runtime.py::layer_resid_p50.
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


def build_jobs(neurons, names, doses):
    jobs = [
        {
            "id": f"baseline-{i}",
            "neuron": None,
            "owner": 0,
            "name": name,
            "prompt": i,
            "dose": 0.0,
        }
        for i, name in enumerate(names)
    ]
    for index, neuron in enumerate(neurons):
        for dose in doses:
            for i, name in enumerate(names):
                jobs.append(
                    {
                        "id": f"n{neuron}-d{dose}-p{i}",
                        "neuron": neuron,
                        "owner": index % 4,
                        "name": name,
                        "prompt": i,
                        "dose": dose,
                    }
                )
    return jobs


def prepare(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    assert torch.cuda.device_count() == 4, "Expected four allocated CUDA GPUs"
    print(
        f"CUDA: {torch.cuda.device_count()} GPUs, {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    config = json.loads(args.prompts.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [
        [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": config["template"].format(name=name)},
        ]
        for name in config["names"]
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        for m in messages
    ]
    prompt_ids = [list(p["input_ids"] if hasattr(p, "keys") else p) for p in prompt_ids]
    if args.stats:
        st = np.load(args.stats)
        assert int(st["layer"]) == args.layer and str(st["model"]) == args.model
        w_down = st["w_down"]
        residual = float(st["resid_norm_q10_50_90"][1])
        calibration = "supplied corpus statistics"
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        lm = model.model.language_model
        assert not getattr(lm.config, "layer_scale", False)
        norms = []

        def record_norm(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms.append(hidden[0, 1:].float().norm(dim=-1).cpu())

        handle = lm.layers[args.layer].register_forward_hook(record_norm)
        ids = tokenizer(REFERENCE_TEXT, return_tensors="pt").input_ids.to("cuda:0")
        with torch.inference_mode():
            lm(input_ids=ids, use_cache=False)
        handle.remove()
        residual = float(norms[0].median())
        w_down = (
            lm.layers[args.layer]
            .mlp.down_proj.weight.detach()
            .float()
            .T.cpu()
            .numpy()
            .astype(np.float16)
        )
        calibration = "explorer reference-text fallback"
    assert 0 < args.neurons <= len(w_down)
    neurons = sorted(
        np.random.default_rng(0)
        .choice(len(w_down), args.neurons, replace=False)
        .tolist()
    )
    np.savez(
        args.output / "directions.npz",
        neurons=neurons,
        w_down=w_down[neurons],
        residual=residual,
    )
    jobs = build_jobs(neurons, config["names"], args.doses)
    manifest = {
        "model": args.model,
        "layer": args.layer,
        "prompts": config,
        "prompt_ids": prompt_ids,
        "jobs": jobs,
        "calibration": calibration,
        "residual_median": residual,
        "max_tokens": 256,
        "temperature": 0,
        "logprobs": 5,
        "thinking": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Prepared {len(jobs)} requests; {len(neurons)} neurons; calibration: {calibration}",
        flush=True,
    )


def generate(args):
    import numpy as np
    import torch.distributed as dist
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    manifest = json.loads((args.output / "manifest.json").read_text())
    jobs = [j for j in manifest["jobs"] if j["owner"] == rank]
    assert jobs, "Select at least four neurons to exercise all DP ranks"
    st = np.load(args.output / "directions.npz")
    directions = dict(
        zip(st["neurons"].tolist(), st["w_down"].astype(np.float32), strict=True)
    )
    llm = LLM(
        model=manifest["model"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        data_parallel_size=4,
        distributed_executor_backend="external_launcher",
        enable_steering=True,
        max_steering_configs=512,
        max_model_len=max(map(len, manifest["prompt_ids"])) + 256 + 16,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = []
    for job in jobs:
        p = copy.deepcopy(llm.get_default_sampling_params())
        p.temperature = 0.0
        p.max_tokens = 256
        p.logprobs = 5
        if job["neuron"] is not None:
            direction = directions[job["neuron"]]
            # Unchanged main_rel formula from steer_sweep.py.
            scale = float(job["dose"] * st["residual"] / np.linalg.norm(direction))
            p.steering_vectors = {
                "mlp_out": {
                    manifest["layer"]: {"vector": direction.tolist(), "scale": scale}
                }
            }
        params.append(p)
    print(f"DP rank {rank}: submitting all {len(jobs)} requests", flush=True)
    outputs = llm.generate(
        [
            TokensPrompt(prompt_token_ids=manifest["prompt_ids"][j["prompt"]])
            for j in jobs
        ],
        params,
        use_tqdm=True,
    )
    with (args.output / f"rank-{rank}.jsonl").open("w") as stream:
        for job, output in zip(jobs, outputs, strict=True):
            answer = output.outputs[0]
            record = {
                **job,
                "layer": manifest["layer"],
                "text": answer.text,
                "token_ids": answer.token_ids,
                "finish": answer.finish_reason,
                "logprobs": [
                    {str(k): v.logprob for k, v in step.items()}
                    for step in answer.logprobs
                ],
            }
            stream.write(json.dumps(record) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def merge(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for rank in range(4)
        for line in (args.output / f"rank-{rank}.jsonl").read_text().splitlines()
    ]
    indexed = {r["id"]: r for r in rows}
    expected = [j["id"] for j in manifest["jobs"]]
    assert len(rows) == len(indexed) == len(expected) and set(indexed) == set(expected)
    assert all(r["text"].strip() for r in rows), "Empty model responses"
    with (args.output / "results.jsonl").open("w") as stream:
        for key in expected:
            stream.write(json.dumps(indexed[key]) + "\n")
    print(
        f"Verified {len(rows)} unique nonempty responses from four DP ranks: {args.output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "generate", "merge"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument(
        "--doses", type=float, nargs="+", default=[-0.8, -0.5, 0.5, 0.8]
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path(__file__).parent / "prompts/lending.json"
    )
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()
    assert args.neurons >= 4
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "generate": generate, "merge": merge}[args.mode](args)


if __name__ == "__main__":
    main()

"""Model layout adapters; steering and analysis formulas stay in their callers."""
import json
import os
from pathlib import Path

import sys

if sys.platform == "darwin":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.8"
    os.environ.setdefault("VLLM_METAL_BACKEND", "mps")

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3-0.6B")


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name):
    if device() == "mps":
        import psutil
        if psutil.virtual_memory().available < 3.5 * 1024**3:
            raise RuntimeError("Not enough free memory for a local model plus system headroom")
        torch.mps.set_per_process_memory_fraction(min(0.9, 3 * 1024**3 / torch.mps.recommended_max_memory()))
    config = AutoConfig.from_pretrained(name)
    cls = AutoModelForCausalLM if config.model_type == "qwen3" else AutoModelForImageTextToText
    model = cls.from_pretrained(name, dtype=torch.bfloat16).to(device()).eval()
    return model, language_model(model)


def language_model(model):
    return getattr(model.model, "language_model", model.model)


def norm_weight(norm):
    weight = norm.weight.float()
    return weight if norm.__class__.__name__ == "Qwen3RMSNorm" else 1.0 + weight


def checkpoint(root):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt") as f:
            index = {key: "model.safetensors" for key in f.keys()}
    prefix = "model.layers" if cfg["model_type"] == "qwen3" else "model.language_model.layers"
    return cfg, index, prefix

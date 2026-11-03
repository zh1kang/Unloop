"""Pinned artifacts and run configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"  # downloaded inputs and large outputs, not tracked

# Hugging Face model ids and commit SHAs (checked 2026-11-02).
MODELS = {
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "released": ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b"),
    "qwen3.5-0.8b-base": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen3.5-2b-base": ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
}
DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
}
# The blog's logs: 200 first samples per benchmark of the unedited and edited models at the full protocol.
BLOG_LOGS = ("RikiyaT/lm_understand-logs", "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f")
BLOG_RUNS = {"qwen3.5-0.8b": "unedited", "released": "p50c"}

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"
# The two tensors the released edit changes (Hugging Face names inside Qwen3_5ForCausalLM).
EDITED_LAYERS = (15, 19)


def load_run_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(path)
    return cfg

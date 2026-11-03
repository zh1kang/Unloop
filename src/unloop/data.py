"""Benchmarks, deterministic subsets, prompts, and the blog's released traces."""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
from dataclasses import dataclass

from .config import BLOG_LOGS, BLOG_RUNS, DATA, DATASETS, INSTRUCTION


@dataclass(frozen=True)
class Problem:
    id: str
    dataset: str
    problem: str
    answer: str


def load_problems(dataset: str) -> list[Problem]:
    from datasets import load_dataset

    repo, rev = DATASETS[dataset]
    if dataset == "math500":
        rows = load_dataset(repo, split="test", revision=rev)
        return [Problem(r["unique_id"], dataset, r["problem"], r["answer"]) for r in rows]
    rows = load_dataset(repo, "main", split="test", revision=rev)
    return [
        Problem(f"gsm8k_test_{i}", dataset, r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
        for i, r in enumerate(rows)
    ]


def blog_traces(model: str, dataset: str) -> list[dict]:
    """The blog's first sample per problem (vLLM, 32,768 tokens): {id, answer, text}, 200 per benchmark."""
    repo, sha = BLOG_LOGS
    run = BLOG_RUNS[model]
    out = []
    for shard in range(10):
        rel = f"exp8_math/eval/{run}/parts/{dataset}_s{shard}of10/{dataset}_first_samples.jsonl.gz"
        path = DATA / "blog" / sha / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}", path)
        out += [json.loads(line) for line in gzip.open(path, "rt")]
    return out


def eval_subset(dataset: str, n: int | None, seed: int, split: str = "heldout") -> list[Problem]:
    """A fixed random subset. "heldout" excludes the problems whose blog traces are used to find directions."""
    problems = load_problems(dataset)
    if split == "heldout":
        used = {t["id"] for t in blog_traces("qwen3.5-0.8b", dataset)}
        problems = [p for p in problems if p.id not in used]
    elif split != "all":
        raise ValueError(f"unknown split {split!r}")
    if n is not None and n < len(problems):
        problems = random.Random(f"{seed}:{dataset}").sample(problems, n)
    return problems


def chat_prompt(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + "\n" + INSTRUCTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

"""Command line: `unloop <command>`. Each command writes a JSON summary under results/."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

from .config import DATA, RESULTS, load_run_config
from .loops import LoopRule

DIRECTIONS = RESULTS / "directions.safetensors"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    log(f"wrote {path}")


def environment() -> dict:
    import torch
    import transformers

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "git": commit or None,
    }


def load_directions() -> dict:
    from safetensors.torch import load_file

    return load_file(DIRECTIONS)


# ---------------------------------------------------------------- conditions


def weight_edits(model, spec: dict, dirs: dict) -> dict:
    """Edited tensors for an `edit` spec: {direction, layers, writers, alpha, top_k}."""
    from .edits import project_out
    from .model import writers

    params = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for layer in spec["layers"]:
        u = dirs[f"{spec['direction']}.u.{layer}"]
        cols = dirs[f"{spec['direction']}.cols.{layer}"][: spec["top_k"]] if spec.get("top_k") else None
        for kind in spec.get("writers", ["mlp"]):
            W = writers(model, layer)[kind].weight
            out[params[id(W)]] = project_out(W.detach().float().cpu(), u, spec.get("alpha", 1.0), cols)
    return out


def apply_condition(model, cond: dict, stack: ExitStack) -> None:
    """Weight edits and forward hooks of a condition; hooks are removed when `stack` closes."""
    from .model import ablate, apply_weight_edits, residual_hooks, steer

    if "edit" not in cond and "hook" not in cond:
        return
    dirs = load_directions()
    if "edit" in cond:
        apply_weight_edits(model, weight_edits(model, cond["edit"], dirs))
    if "hook" in cond:
        h = cond["hook"]
        layers = h["layers"] if h["layers"] != "all" else list(range(len(model.model.layers)))
        for layer in layers:
            if h["kind"] == "steer":  # move by `scale` times the raw loop-minus-productive mean difference
                r = dirs[f"{h['direction']}.r.{layer}"]
                fn = steer(r / r.norm(), h["scale"] * float(r.norm()))
            else:
                fn = ablate(dirs[f"{h['direction']}.u.{layer}"])
            stack.enter_context(residual_hooks(model, fn, at=[layer]))


def problems_for(cfg: dict):
    from .data import eval_subset

    ev = cfg["eval"]
    for dataset, n in ev["datasets"].items():
        for p in eval_subset(dataset, None if n == "all" else n, ev["seed"], ev.get("split", "heldout")):
            for k in range(ev.get("samples", {}).get(dataset, 1)):
                yield p, k


# ---------------------------------------------------------------- eval


def score(rec: dict, problem, tok, rule: LoopRule) -> dict:
    from .grading import extract_answer, is_correct
    from .loops import loop_onset

    text = tok.decode(rec["tokens"], skip_special_tokens=False)
    pred = extract_answer(text)
    onset = loop_onset(rec["tokens"], rule)
    return {
        "id": problem.id,
        "dataset": problem.dataset,
        "n_tokens": len(rec["tokens"]),
        "finish": rec["finish"],
        "forced": rec["forced"],
        "think_closed": "</think>" in text,
        "pred": pred,
        "correct": is_correct(pred, problem.answer),
        "loop": onset is not None,
        "loop_onset": onset,
        "text": text,
    }


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        n = len(rs)
        m = {"n": n}
        for key in ("correct", "loop", "think_closed"):
            k = sum(r[key] for r in rs)
            m[key] = round(k / n, 4)
            m[key + "_ci95"] = wilson(k, n)
        m["cut"] = round(sum(r["finish"] == "length" for r in rs) / n, 4)
        m["mean_tokens"] = round(sum(r["n_tokens"] for r in rs) / n, 1)
        out[ds] = m
    return out


def cmd_eval(args) -> None:
    import torch

    from .data import chat_prompt
    from .generate import Decoding, Sampling, generate_hf, generate_vllm
    from .model import load

    cfg = load_run_config(args.config)
    cond = cfg["conditions"][args.condition]
    sampling = replace(Sampling(**cfg["sampling"]), **cond.get("sampling", {}))
    decoding = Decoding(**cond.get("decoding", {}))
    rule = LoopRule(**cfg.get("loop", {}))
    backend = args.backend or cond.get("backend", cfg.get("backend", "hf"))
    items = list(problems_for(cfg))
    log(f"{args.condition}: {len(items)} generations, backend {backend}, {sampling}, {decoding}")

    t0 = time.time()
    if backend == "hf":
        model, tok = load(cond.get("model", "qwen3.5-0.8b"))
        prompts = [chat_prompt(tok, p.problem) for p, _ in items]
        with ExitStack() as stack:
            apply_condition(model, cond, stack)
            torch.manual_seed(sampling.seed)
            recs = generate_hf(model, tok, prompts, sampling, decoding, cfg.get("batch_size", 16), log)
    else:
        path = vllm_checkpoint(cond, args.condition)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        prompts = [chat_prompt(tok, p.problem) for p, _ in items]
        recs = generate_vllm(path, tok, prompts, sampling, decoding)
    seconds = time.time() - t0

    rows = [score(rec, p, tok, rule) | {"sample": k} for rec, (p, k) in zip(recs, items)]
    run = DATA / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    with open(run / f"{args.condition}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    write_json(
        RESULTS / cfg["name"] / f"{args.condition}.json",
        {
            "condition": args.condition,
            "spec": cond,
            "backend": backend,
            "sampling": asdict(sampling),
            "decoding": asdict(decoding),
            "loop_rule": asdict(rule),
            "eval": cfg["eval"],
            "seconds": round(seconds),
            "batch_size": cfg.get("batch_size", 16),
            "env": environment(),
            "metrics": summarize(rows),
            "per_sample": [{k: v for k, v in r.items() if k != "text"} for r in rows],
        },
    )


def vllm_checkpoint(cond: dict, name: str) -> str:
    """A model path vLLM can load: the pinned repo, or a saved text-only checkpoint with the condition's edits."""
    from huggingface_hub import snapshot_download

    from .config import MODELS
    from .model import load

    if "hook" in cond:
        raise NotImplementedError("forward hooks need the hf backend")
    repo, rev = MODELS[cond.get("model", "qwen3.5-0.8b")]
    if "edit" not in cond:
        return snapshot_download(repo, revision=rev)
    out = DATA / "checkpoints" / name
    if not (out / "config.json").exists():
        model, tok = load(cond.get("model", "qwen3.5-0.8b"), dev="cpu")
        with ExitStack() as stack:
            apply_condition(model, cond, stack)
            model.save_pretrained(out)
        tok.save_pretrained(out)
    return str(out)


# ---------------------------------------------------------------- table


def paired_diff(a: list[dict], b: list[dict], key: str, reps: int = 2000) -> list[float]:
    """Mean of b - a over matched samples, with a 95% bootstrap interval over problems."""
    ka = {(r["id"], r["sample"]): r[key] for r in a}
    pairs = [(float(ka[(r["id"], r["sample"])]), float(r[key])) for r in b if (r["id"], r["sample"]) in ka]
    diffs = [y - x for x, y in pairs]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps))
    return [round(sum(diffs) / len(diffs), 4), round(boots[int(0.025 * reps)], 4), round(boots[int(0.975 * reps)], 4)]


def cmd_table(args) -> None:
    folder = RESULTS / args.run
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if p.stem != "table"}
    base = runs.get(args.baseline)
    datasets = sorted({ds for r in runs.values() for ds in r["metrics"]})
    lines = [
        "| condition | " + " | ".join(f"{ds} acc | {ds} loop | {ds} cut" for ds in datasets) + " |",
        "|---|" + "---|---|---|" * len(datasets),
    ]
    table = {}
    for name, r in runs.items():
        cells, entry = [], {}
        for ds in datasets:
            m = r["metrics"].get(ds)
            if not m:
                cells += ["-"] * 3
                continue
            entry[ds] = dict(m)
            if base and name != args.baseline:
                a = [x for x in base["per_sample"] if x["dataset"] == ds]
                b = [x for x in r["per_sample"] if x["dataset"] == ds]
                entry[ds]["d_correct"] = paired_diff(a, b, "correct")
                entry[ds]["d_loop"] = paired_diff(a, b, "loop")
            cells += [f"{100 * m['correct']:.1f}", f"{100 * m['loop']:.1f}", f"{100 * m['cut']:.1f}"]
        table[name] = entry
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    write_json(folder / "table.json", {"baseline": args.baseline, "rows": table, "markdown": "\n".join(lines)})


def main() -> None:
    from . import analysis

    ap = argparse.ArgumentParser(prog="unloop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval", help="generate and grade one condition of a run config")
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--backend", choices=["hf", "vllm"])
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("table", help="results table of a run with paired differences to a baseline")
    p.add_argument("--run", default="pilot")
    p.add_argument("--baseline", default="base")
    p.set_defaults(fn=cmd_table)
    analysis.add_commands(sub)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

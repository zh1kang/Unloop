"""Analyses that read weights and activations (teacher-forced, no sampling).

reproduce   diff the released weights against the base, recover the edit, loop rates of the blog's 32k traces
directions  difference-of-means loop direction per layer from the base model's own looping traces
sweep       teacher-forced proxy for many edits and steering settings; selects the ones to generate with
explain     logit lens of the directions, the </think> hypothesis, projections over time
origin      the same direction in Qwen3.5-0.8B-Base and the 2B models
"""

from __future__ import annotations

import hashlib
import statistics
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from .conditions import SWEEP, apply_condition, load_directions, save_directions
from .config import EDITED_LAYERS, MODELS, RESULTS, THINK_END
from .loops import LoopRule, first_stuck, loop_onset

GAP, WIN = 64, 256  # productive window [onset-GAP-WIN, onset-GAP), loop window [onset+GAP, onset+GAP+WIN)


@dataclass
class Trace:
    id: str
    dataset: str
    prompt: list[int]
    response: list[int]
    onset: int | None
    finished: bool

    def windows(self) -> dict[str, list[int]]:
        """Absolute positions of the productive and loop windows of a looping trace."""
        o, p = self.onset, len(self.prompt)
        return {
            "prod": list(range(p + o - GAP - WIN, p + o - GAP)),
            "loop": list(range(p + o + GAP, p + o + GAP + WIN)),
        }

    def ids(self, end: int | None = None) -> list[int]:
        return (self.prompt + self.response)[:end]


def split_of(problem_id: str) -> str:
    return "fit" if hashlib.sha1(problem_id.encode()).digest()[0] % 2 == 0 else "val"


def load_traces(model: str, tok, rule: LoopRule, datasets=("gsm8k", "math500")) -> list[Trace]:
    from .data import blog_traces, chat_prompt, load_problems

    out = []
    for ds in datasets:
        problems = {p.id: p for p in load_problems(ds)}
        for t in blog_traces(model, ds):
            resp = tok(t["text"], add_special_tokens=False).input_ids
            prompt = tok(chat_prompt(tok, problems[t["id"]].problem), add_special_tokens=False).input_ids
            out.append(Trace(t["id"], ds, prompt, resp, loop_onset(resp, rule), len(resp) < 32760))
    return out


def usable(traces: list[Trace], split: str, max_len: int, limit: int) -> list[Trace]:
    """Looping traces whose two windows fit, shortest first, from one split."""
    ok = [
        t
        for t in traces
        if t.onset is not None
        and split_of(t.id) == split
        and t.onset >= GAP + WIN
        and t.onset + GAP + WIN <= len(t.response)
        and len(t.prompt) + t.onset + GAP + WIN <= max_len
    ]
    return sorted(ok, key=lambda t: (t.onset, t.id))[:limit]


@torch.no_grad()
def capture(model, ids: list[int], sets: dict[str, list[int]], mlp: bool = True) -> dict:
    """Mean residual stream (after every layer) and mean down_proj input over each set of positions."""
    from .model import layers, residual_hooks

    dev = model.device
    idx = {k: torch.tensor(v, device=dev) for k, v in sets.items()}
    n = len(layers(model))
    res = {k: [None] * n for k in sets}
    act = {k: [None] * n for k in sets}

    def on_res(i, h):
        for k, ix in idx.items():
            res[k][i] = h[0, ix].float().mean(0).cpu()

    def on_mlp(i):
        def hook(_m, args):
            for k, ix in idx.items():
                act[k][i] = args[0][0, ix].float().mean(0).cpu()

        return hook

    handles = [layers(model)[i].mlp.down_proj.register_forward_pre_hook(on_mlp(i)) for i in range(n)] if mlp else []
    try:
        with residual_hooks(model, on_res):
            model.model(input_ids=torch.tensor([ids], device=dev))
    finally:
        for h in handles:
            h.remove()
    out = {k: torch.stack(v) for k, v in res.items()}
    if mlp:
        out |= {f"mlp_{k}": torch.stack(v) for k, v in act.items()}
    return out


@torch.no_grad()
def next_token_stats(model, t: Trace, think_id: int) -> dict:
    """Mean log p of the next token and of </think> in both windows; share of positions with </think> in the top 20."""
    sets = t.windows()
    pred = sorted(p - 1 for v in sets.values() for p in v)  # logits at p - 1 predict token p
    ids = t.ids(max(pred) + 2)
    keep = torch.tensor(pred, device=model.device)
    logits = model(input_ids=torch.tensor([ids], device=model.device), logits_to_keep=keep).logits[0]
    lp = logits.float().log_softmax(-1)
    row = {p: i for i, p in enumerate(pred)}
    out = {}
    for k, v in sets.items():
        r = torch.tensor([row[p - 1] for p in v])
        tgt = torch.tensor([ids[p] for p in v], device=lp.device)
        out[f"{k}_logp"] = float(lp[r].gather(-1, tgt[:, None]).mean())
        out[f"{k}_think_logp"] = float(lp[r, think_id].mean())
        out[f"{k}_think_top20"] = float((lp[r] > lp[r, think_id][:, None]).sum(-1).lt(20).float().mean())
    return out


def mean_dicts(rows: list[dict]) -> dict:
    return {k: round(statistics.fmean(r[k] for r in rows), 5) for k in rows[0]}


# ---------------------------------------------------------------- reproduce


def released_delta() -> tuple[dict, dict]:
    """Bitwise diff of the released checkpoint against the base; rank-1 recovery of each changed tensor."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from .edits import changed, fit_alpha, project_out, rank1

    files = []
    for name in ("qwen3.5-0.8b", "released"):
        repo, rev = MODELS[name]
        files.append(hf_hub_download(repo, "model.safetensors-00001-of-00001.safetensors", revision=rev))
    report, dirs = {"tensors": {}}, {}
    with safe_open(files[0], "pt") as fb, safe_open(files[1], "pt") as fe:
        assert set(fb.keys()) == set(fe.keys())
        report["n_tensors"] = len(fb.keys())
        report["n_values"] = sum(_numel(fb, k) for k in fb.keys())
        for key in sorted(fb.keys()):
            W0, W1 = fb.get_tensor(key), fe.get_tensor(key)
            if torch.equal(W0, W1):
                continue
            info = changed(W0, W1)
            u, s1, ratio = rank1(W1.double() - W0.double())
            alpha = fit_alpha(W0, W1, u, info["columns"])
            rebuilt = project_out(W0.float(), u, alpha, info["columns"]).to(torch.bfloat16)
            info |= {
                "sigma1": s1,
                "sigma2_over_sigma1": ratio,
                "alpha": alpha,
                "rebuilt_bitwise_equal_share": float((rebuilt == W1).float().mean()),
            }
            info["top64_by_weight_overlap"] = len(
                set(info["columns"]) & set((u @ W0.float()).abs().topk(64).indices.tolist())
            )
            layer = int(key.split(".layers.")[1].split(".")[0])
            dirs |= {
                f"released.u.{layer}": u,
                f"released.cols.{layer}": torch.tensor(info["columns"]),
                f"released.alpha.{layer}": torch.tensor(alpha),
            }
            report["tensors"][key] = info
    report["changed_values"] = sum(t["changed_entries"] for t in report["tensors"].values())
    us = [dirs[f"released.u.{L}"] for L in EDITED_LAYERS]
    report["cos_u15_u19"] = float(us[0] @ us[1])
    return report, dirs


def _numel(f, key: str) -> int:
    n = 1
    for d in f.get_slice(key).get_shape():
        n *= d
    return n


def blog_loop_stats(tok, rule: LoopRule) -> dict:
    """Loop statistics of the blog's released first samples (full protocol, 32,768 tokens)."""
    from .data import blog_traces
    from .grading import extract_answer, is_correct

    out = {}
    for model in ("qwen3.5-0.8b", "released"):
        for ds in ("gsm8k", "math500"):
            rows = []
            for t in blog_traces(model, ds):
                ids = tok(t["text"], add_special_tokens=False).input_ids
                stop = first_stuck(ids, rule)
                rows.append(
                    {
                        "correct": is_correct(extract_answer(t["text"]), t["answer"]),
                        "cut": len(ids) >= 32760,
                        "loop": loop_onset(ids, rule),
                        "stop": stop,
                    }
                )
            loops = sorted(r["loop"] for r in rows if r["loop"] is not None)
            fired = [r for r in rows if r["stop"] is not None]
            out[f"{model}/{ds}"] = {
                "n": len(rows),
                "correct": _share(rows, "correct"),
                "cut": _share(rows, "cut"),
                "loop": len(loops) / len(rows),
                "loop_onset_quartiles": _quartiles(loops),
                "early_stop_fired": len(fired),
                "early_stop_but_finished": sum(not r["cut"] for r in fired),
                "early_stop_but_correct": sum(r["correct"] for r in fired),
            }
    return out


def _share(rows, key):
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4)


def _quartiles(xs):
    return [xs[len(xs) // 4], xs[len(xs) // 2], xs[3 * len(xs) // 4]] if xs else None


def cmd_reproduce(args) -> None:
    from transformers import AutoTokenizer

    from .cli import write_json

    report, dirs = released_delta()
    save_directions(dirs)
    repo, rev = MODELS["qwen3.5-0.8b"]
    report["blog_traces"] = blog_loop_stats(AutoTokenizer.from_pretrained(repo, revision=rev), LoopRule())
    write_json(RESULTS / "reproduce.json", report)


# ---------------------------------------------------------------- directions


def fit_directions(model, traces: list[Trace]) -> tuple[dict, dict]:
    """Per-layer loop-minus-productive mean differences, per-layer separation, and MLP writer rankings."""
    from .edits import top_writers, unit
    from .model import writers

    caps = [capture(model, t.ids(t.windows()["loop"][-1] + 1), t.windows()) for t in traces]
    stack = {k: torch.stack([c[k] for c in caps]) for k in caps[0]}  # [traces, layers, d]
    r = stack["loop"].mean(0) - stack["prod"].mean(0)
    dirs, stats = {}, {"n_traces": len(traces), "layers": []}
    old = load_directions_safe()
    for L in range(r.shape[0]):
        u = unit(r[L])
        proj = (stack["loop"][:, L] - stack["prod"][:, L]) @ u  # per-trace paired projection difference
        W = writers(model, L)["mlp"].weight.detach().float().cpu()
        act = stack["mlp_loop"][:, L].mean(0) - stack["mlp_prod"][:, L].mean(0)
        dirs |= {f"diffmean.r.{L}": r[L], f"diffmean.u.{L}": u, f"diffmean.cols.{L}": _ranked(act, W, u, 256)}
        row = {"layer": L, "norm": float(r[L].norm()), "paired_d": float(proj.mean() / proj.std())}
        if f"released.u.{L}" in old:
            ru = old[f"released.u.{L}"]
            sign = 1.0 if float(r[L] @ ru) >= 0 else -1.0  # orient: the released direction is higher in loops
            dirs[f"released.u.{L}"] = sign * ru
            cols = old[f"released.cols.{L}"].tolist()
            row |= {
                "cos_released": abs(float(u @ ru)),
                "released_cols_in_our_top64": len(set(cols) & set(top_writers(act, W, sign * ru, 64))),
                "released_cols_in_diffmean_top64": len(set(cols) & set(dirs[f"diffmean.cols.{L}"][:64].tolist())),
            }
        for E in EDITED_LAYERS:
            if f"released.u.{E}" in old:
                row[f"cos_released_u{E}"] = abs(float(u @ old[f"released.u.{E}"]))
        stats["layers"].append(row)
    return dirs, stats


def _ranked(act, W, u, k):
    from .edits import unit

    score = act.double() * (unit(u.double()) @ W.double())
    return score.topk(min(k, score.numel())).indices


def load_directions_safe() -> dict:
    from .conditions import DIRECTIONS

    return load_directions() if DIRECTIONS.exists() else {}


def cmd_directions(args) -> None:
    from .cli import log, write_json
    from .model import load

    model, tok = load("qwen3.5-0.8b")
    traces = usable(load_traces("qwen3.5-0.8b", tok, LoopRule()), "fit", args.max_len, args.max_traces)
    log(f"fitting on {len(traces)} looping traces")
    dirs, stats = fit_directions(model, traces)
    save_directions(dirs)
    stats["trace_ids"] = [t.id for t in traces]
    write_json(RESULTS / "directions.json", stats)


# ---------------------------------------------------------------- sweep


def candidates(n_layers: int) -> dict[str, dict]:
    c = {
        "none": {},
        "released": {"edit": {"direction": "released", "layers": list(EDITED_LAYERS), "alpha": "fit", "top_k": 64}},
    }
    for L in range(n_layers):
        c[f"ablate_mlp_{L}"] = {"edit": {"direction": "diffmean", "layers": [L]}}
    c["ablate_mlp_15_19"] = {"edit": {"direction": "diffmean", "layers": list(EDITED_LAYERS)}}
    c["released_form_diffmean"] = {
        "edit": {"direction": "diffmean", "layers": list(EDITED_LAYERS), "alpha": 6.0, "top_k": 64}
    }
    c["ablate_all_writers"] = {
        "edit": {"direction": "diffmean", "layers": list(range(n_layers)), "writers": ["mlp", "attn"]}
    }
    c["ablate_residual_all"] = {"hook": {"kind": "ablate", "direction": "diffmean", "layers": "all"}}
    for L in (7, 11, 15, 19):
        for s in (-1.0, -2.0, -4.0):
            c[f"steer_{L}_{s:g}"] = {"hook": {"kind": "steer", "direction": "diffmean", "layers": [L], "scale": s}}
    return c


def select(results: dict, max_prod_drop: float) -> dict:
    """Largest drop in loop-window log p among candidates that keep the productive window within max_prod_drop."""
    best = {}
    for key, prefix in (("edit", ("ablate", "released_form")), ("hook", ("steer", "ablate_residual"))):
        ok = {n: r for n, r in results.items() if n.startswith(prefix) and r["d_prod_logp"] >= -max_prod_drop}
        if ok:
            name = min(ok, key=lambda n: ok[n]["d_loop_logp"])
            best[key] = ok[name]["spec"][key]
            best[f"{key}_name"] = name
    return best


def cmd_sweep(args) -> None:
    from .cli import log, write_json
    from .model import load

    model, tok = load("qwen3.5-0.8b")
    think_id = tok.convert_tokens_to_ids(THINK_END)
    traces = usable(load_traces("qwen3.5-0.8b", tok, LoopRule()), "val", args.max_len, args.max_traces)
    results = {}
    for name, spec in candidates(len(model.model.layers)).items():
        with ExitStack() as stack:
            apply_condition(model, spec, stack)
            results[name] = mean_dicts([next_token_stats(model, t, think_id) for t in traces]) | {"spec": spec}
        if name != "none":
            for k in ("prod_logp", "loop_logp", "loop_think_logp"):
                results[name][f"d_{k}"] = round(results[name][k] - results["none"][k], 5)
        log(f"{name}: {results[name]}")
    results["none"] |= {"d_prod_logp": 0.0, "d_loop_logp": 0.0}
    write_json(
        SWEEP,
        {
            "n_traces": len(traces),
            "max_prod_drop": args.max_prod_drop,
            "best": select(results, args.max_prod_drop),
            "candidates": results,
        },
    )


# ---------------------------------------------------------------- explain


def lens(model, tok, u: torch.Tensor, k: int = 15) -> dict:
    from .edits import logit_lens
    from .model import unembed

    W_U, gain = unembed(model)
    x = logit_lens(u.to(W_U.device), W_U, gain.to(W_U.device)).float().cpu()
    z = (x - x.mean()) / x.std()
    order = x.argsort()
    special = {t: tok.convert_tokens_to_ids(t) for t in (THINK_END, "<|im_end|>", "<think>")}
    return {
        "promoted": [tok.decode([i]) for i in order[-k:].flip(0).tolist()],
        "suppressed": [tok.decode([i]) for i in order[:k].tolist()],
        "special": {
            t: {"z": round(float(z[i]), 2), "rank_from_bottom": int((x < x[i]).sum())} for t, i in special.items()
        },
    }


@torch.no_grad()
def projection_curve(model, ids: list[int], u: torch.Tensor, layer: int, start: int) -> list[float]:
    """Per-token projection of the residual stream after `layer` onto u, for positions >= start."""
    from .model import residual_hooks

    out = {}
    with residual_hooks(
        model, lambda i, h: out.__setitem__("h", h[0, start:].float() @ u.to(h.device).float()), at=[layer]
    ):
        model.model(input_ids=torch.tensor([ids], device=model.device))
    return out["h"].cpu().tolist()


def binned(xs: list[float], width: int = 64) -> list[float]:
    return [round(statistics.fmean(xs[i : i + width]), 4) for i in range(0, len(xs) - width + 1, width)]


def cmd_explain(args) -> None:
    from .cli import write_json
    from .model import load

    model, tok = load("qwen3.5-0.8b")
    dirs = load_directions()
    report = {
        "lens": {
            f"{s}.{L}": lens(model, tok, dirs[f"{s}.u.{L}"]) for s in ("released", "diffmean") for L in EDITED_LAYERS
        }
    }
    rule = LoopRule()
    loops = usable(load_traces("qwen3.5-0.8b", tok, rule), "val", args.max_len, args.max_traces)
    clean = [t for t in load_traces("released", tok, rule) if t.onset is None and t.finished][: args.max_traces]
    curves = {}
    for L in EDITED_LAYERS:
        u = dirs[f"released.u.{L}"]
        before, after, loop_bins, clean_bins = [], [], [], []
        for t in loops:  # aligned at the loop onset: 1,024 tokens either side
            s = len(t.prompt) + max(0, t.onset - 1024)
            c = projection_curve(model, t.ids(len(t.prompt) + t.onset + 1024), u, L, s)
            pre = t.onset - max(0, t.onset - 1024)
            before += c[:pre]
            after += c[pre:]
            loop_bins.append(binned([float("nan")] * (1024 - pre) + c))
        for t in clean:  # text of the released model that finished without a loop, read by the base model
            c = projection_curve(model, t.ids(args.max_len), u, L, len(t.prompt))
            clean_bins.append(binned(c))
        curves[str(L)] = {
            "loop_before_onset": statistics.fmean(before),
            "loop_after_onset": statistics.fmean(after),
            "non_looping": statistics.fmean(x for b in clean_bins for x in b) if clean_bins else None,
            "loop_bins_aligned": loop_bins,
            "non_looping_bins": clean_bins,
        }
    report["projection"] = curves
    think_id = tok.convert_tokens_to_ids(THINK_END)
    report["think"] = {}
    for name, spec in (("base", {}), ("released", candidates(0)["released"])):
        with ExitStack() as stack:
            apply_condition(model, spec, stack)
            report["think"][name] = mean_dicts([next_token_stats(model, t, think_id) for t in loops])
    report["n_loop_traces"], report["n_clean_traces"] = len(loops), len(clean)
    write_json(RESULTS / "explain.json", report)
    plot_projection(curves, RESULTS / "figures" / "projection.png")


def plot_projection(curves: dict, path) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 3.2), squeeze=False)
    for ax, (L, c) in zip(axes[0], curves.items()):
        for name, bins, x0, color in (
            ("looping (aligned at onset)", c["loop_bins_aligned"], -1024, "C3"),
            ("non-looping", c["non_looping_bins"], 0, "C0"),
        ):
            width = max((len(b) for b in bins), default=0)
            mean = [
                statistics.fmean([b[i] for b in bins if i < len(b) and not math.isnan(b[i])] or [math.nan])
                for i in range(width)
            ]
            ax.plot([x0 + 64 * i for i in range(width)], mean, color=color, label=name)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("tokens (looping: relative to loop onset)")
        ax.set_ylabel(f"projection on released u, layer {L}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- origin


def cmd_origin(args) -> None:
    from .cli import log, write_json
    from .edits import cosine
    from .model import load

    ref_tok = load_tok("qwen3.5-0.8b")
    traces = usable(load_traces("qwen3.5-0.8b", ref_tok, LoopRule()), "fit", args.max_len, args.max_traces)
    dirs = load_directions()
    report = {}
    for name in args.models:
        model, tok = load(name)
        if tok.get_vocab() != ref_tok.get_vocab():
            raise ValueError(f"{name} has a different vocabulary; the traces cannot be reused")
        ds, stats = fit_directions(model, traces)
        n = len(model.model.layers)
        rows = []
        for L in range(n):
            row = {"layer": L, "depth": round(L / (n - 1), 3), "paired_d": stats["layers"][L]["paired_d"]}
            ref = dirs.get(f"diffmean.r.{L}")
            if ref is not None and ref.shape == ds[f"diffmean.r.{L}"].shape:  # same width: same residual basis?
                row["cos_with_0.8b"] = cosine(ds[f"diffmean.r.{L}"], ref)
                if f"released.u.{L}" in dirs:
                    row["cos_with_released"] = cosine(ds[f"diffmean.r.{L}"], dirs[f"released.u.{L}"])
            rows.append(row)
        best = max(rows, key=lambda r: r["paired_d"])
        report[name] = {
            "layers": rows,
            "best_layer": best["layer"],
            "lens_best": lens(model, tok, ds[f"diffmean.u.{best['layer']}"]),
        }
        log(f"{name}: best layer {best['layer']} paired d {best['paired_d']:.2f}")
        del model
    write_json(RESULTS / "origin.json", {"n_traces": len(traces), "models": report})


def load_tok(name: str):
    from transformers import AutoTokenizer

    repo, rev = MODELS[name]
    return AutoTokenizer.from_pretrained(repo, revision=rev)


def add_commands(sub) -> None:
    sub.add_parser("reproduce", help="weight diff of the released edit and loop rates of the blog traces").set_defaults(
        fn=cmd_reproduce
    )
    for name, fn, help_ in (
        ("directions", cmd_directions, "difference-of-means loop directions"),
        ("sweep", cmd_sweep, "teacher-forced sweep of edits and steering"),
        ("explain", cmd_explain, "logit lens, </think> effect, projections over time"),
        ("origin", cmd_origin, "the loop direction in other Qwen3.5 models"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--max-traces", type=int, default=64)
        p.add_argument("--max-len", type=int, default=8192, help="longest teacher-forced context")
        p.set_defaults(fn=fn)
        if name == "sweep":
            p.add_argument("--max-prod-drop", type=float, default=0.02, help="nats/token allowed on productive text")
        if name == "origin":
            p.add_argument("--models", nargs="+", default=["qwen3.5-0.8b-base", "qwen3.5-2b", "qwen3.5-2b-base"])

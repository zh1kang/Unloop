"""A rank-1 weight edit trained to leave a loop, then measured by restarting generation.

The sweep's teacher-forced log-probability barely moves under the released edit, so it cannot
choose one. Here the parameters are the edit itself. On two MLP down-projections (layers 15 and
19, the layers the released edit uses):

    W <- W + u v^T

with u a unit vector in the residual stream and v a gain per neuron. u starts as the
difference-of-means direction and v starts near zero, so the fit can leave that family. The loss
on each looping fit trace is a pair: next-token loss on the productive window before the loop,
and unlikelihood of the repeated tokens through the next 1,024 tokens of the loop. With
`--hold-think`, that loop stretch also pays the log-probability of `</think>`, so ending the
trace is not how the repeat dies. The same next-token loss on finished, non-looping fit traces
is the control that ordinary text stays likely. Nothing is trained on the released weights.

The measurement is the branch-point restart, not this loss. `unloop branch` continues the same
32 validation prefixes as the pilot (configs/branch_val.json): 8 samples, 4,096 new tokens, stop
at </think>. A continuation counts as looping only when thinking never closes and the tail is a
loop. Base and released numbers from that pilot are the comparison.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .analysis import GAP, WIN, Trace, load_traces, split_of
from .config import RESULTS, ROOT, THINK_END
from .edits import cosine, unit
from .loops import LoopRule, loop_onset, repeated_mask

CONTEXT, BREAK_LEN, MAX_LEN = 512, 1024, 4096


def escape_pair_loss(
    log_probs: torch.Tensor, targets: torch.Tensor, break_mask: torch.Tensor, hold_id: int | None = None
) -> torch.Tensor:
    """Mean next-token loss on keep positions plus mean unlikelihood on break positions.

    Unlikelihood is -log(1 - p) of the token that continues a repeated n-gram (Welleck et al.).
    `hold_id` adds the mean log-probability of that token on break positions, which is higher
    when the token is likely.
    """
    return sum(loss_parts(log_probs, targets, break_mask, hold_id).values())


def loss_parts(
    log_probs: torch.Tensor, targets: torch.Tensor, break_mask: torch.Tensor, hold_id: int | None = None
) -> dict[str, torch.Tensor]:
    lp = log_probs.gather(1, targets[:, None]).squeeze(1)
    parts = {}
    keep = ~break_mask
    if keep.any():
        parts["keep"] = (-lp[keep]).mean()
    if break_mask.any():
        p = lp[break_mask].exp().clamp(max=1 - 1e-4)
        parts["break"] = (-torch.log(1 - p)).mean()
        if hold_id is not None:
            parts["hold"] = log_probs[break_mask, hold_id].mean()
    if not parts:
        raise ValueError("crop has no scored positions")
    return parts


def pair_crop(
    trace: Trace,
    context: int = CONTEXT,
    break_len: int = BREAK_LEN,
    max_len: int = MAX_LEN,
    rule: LoopRule = LoopRule(),
) -> dict | None:
    """One looping trace cropped to the productive window and the repeated tokens of the loop.

    `start` is the absolute position of ids[0], so RoPE can keep the original positions when the
    prompt no longer fits in max_len.
    """
    if trace.onset is None or trace.onset < GAP + WIN:
        return None
    full = trace.prompt + trace.response
    p, o = len(trace.prompt), trace.onset
    keep = list(range(p + o - GAP - WIN, p + o - GAP))
    if keep[0] < 0 or keep[-1] >= len(full):
        return None
    mask = repeated_mask(trace.response, rule.n)
    brk = [i for i in range(p + o, min(len(full), p + o + break_len)) if mask[i - p]]
    if len(brk) < 32:
        return None
    end = brk[-1] + 1
    start = max(0, keep[0] - context)
    if end - start > max_len:
        start = end - max_len
    if start > keep[0] - 1:
        return None
    scored = [(t, False) for t in keep] + [(t, True) for t in brk]
    return {
        "ids": full[start:end],
        "start": start,
        "index": [t - 1 - start for t, _ in scored],
        "targets": [full[t] for t, _ in scored],
        "break": [flag for _, flag in scored],
    }


def clean_crop(trace: Trace, context: int = CONTEXT, win: int = WIN, max_len: int = MAX_LEN) -> dict | None:
    """A window of ordinary text from a finished, non-looping trace. Every position is a keep."""
    if trace.onset is not None or not trace.finished or len(trace.response) < win:
        return None
    full = trace.prompt + trace.response
    keep = list(range(len(full) - win, len(full)))
    end = keep[-1] + 1
    start = max(0, keep[0] - context)
    if end - start > max_len:
        start = end - max_len
    if start > keep[0] - 1:
        return None
    return {
        "ids": full[start:end],
        "start": start,
        "index": [t - 1 - start for t in keep],
        "targets": [full[t] for t in keep],
        "break": [False] * len(keep),
    }


def continuation_outcome(prefix: list[int], new: list[int], think_id: int, stop: set[int], rule: LoopRule) -> dict:
    """The pilot's labels: thinking closed, or still going and the tail is a loop."""
    closed = think_id in new or any(t in stop for t in new)
    return {
        "closed": closed,
        "looping": (not closed) and loop_onset(prefix + new, rule) is not None,
        "n_new": len(new),
    }


class Rank1Write(nn.Module):
    """u and v for W <- W + u v^T on mlp.down_proj at each edited layer."""

    def __init__(self, layers: list[int], intermediate: int, u_init: dict[int, torch.Tensor], seed: int = 0):
        super().__init__()
        self.layers = list(layers)
        g = torch.Generator().manual_seed(seed)
        self.u = nn.ParameterDict({str(L): nn.Parameter(u_init[L].detach().float().clone()) for L in layers})
        self.v = nn.ParameterDict(
            {str(L): nn.Parameter(1e-4 * torch.randn(intermediate, generator=g)) for L in layers}
        )

    def direction(self, layer: int) -> torch.Tensor:
        u = self.u[str(layer)]
        return u / u.norm().clamp(min=1e-8)

    @contextmanager
    def applied(self, model):
        """Add the edit inside down_proj. Backward starts at the first edited layer, not at the embedding."""
        from .model import layers

        handles = []

        def detach(_module, args):
            return (args[0].detach(),) + args[1:]

        handles.append(layers(model)[min(self.layers)].register_forward_pre_hook(detach))

        def hook_for(layer: int):
            def hook(_module, args, output):
                u = self.direction(layer).to(device=output.device, dtype=torch.float32)
                v = self.v[str(layer)].to(device=output.device, dtype=torch.float32)
                gain = args[0].detach().float() @ v
                return output + (gain.unsqueeze(-1) * u).to(dtype=output.dtype)

            return hook

        for layer in self.layers:
            handles.append(layers(model)[layer].mlp.down_proj.register_forward_hook(hook_for(layer)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def logp_at(model, crop: dict) -> torch.Tensor:
    dev = next(model.parameters()).device
    ids = torch.tensor([crop["ids"]], device=dev)
    pos = torch.arange(crop["start"], crop["start"] + len(crop["ids"]), device=dev).unsqueeze(0)
    index = torch.tensor(crop["index"], device=dev)
    logits = model(input_ids=ids, position_ids=pos, logits_to_keep=index, use_cache=False).logits[0]
    return logits.float().log_softmax(-1)


def sequence_loss(model, crop: dict, hold_id: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward. The scalar sums the part means; the dict keeps each part for a backward of its own."""
    dev = next(model.parameters()).device
    targets = torch.tensor(crop["targets"], device=dev)
    mask = torch.tensor(crop["break"], device=dev)
    parts = loss_parts(logp_at(model, crop), targets, mask, hold_id)
    return sum(parts.values()), parts


def _backward_equal(module: nn.Module, terms: list[torch.Tensor]) -> None:
    """Give each term the same share of the step.

    The repeat unlikelihood starts with a huge gradient, and a plain sum then a global clip spends
    the whole step on it. </think> is rare at the start, so its penalty would otherwise vanish.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    acc = {id(p): torch.zeros_like(p) for p in params}
    live = [t for t in terms if t.requires_grad]
    for i, term in enumerate(live):
        module.zero_grad()
        term.backward(retain_graph=i < len(live) - 1)
        square = sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None)
        if square < 1e-12:
            continue
        scale = square**-0.5
        for p in params:
            if p.grad is not None:
                acc[id(p)].add_(p.grad, alpha=scale)
    for p in params:
        p.grad = acc[id(p)] / max(len(live), 1)


def _unit_grad(module: nn.Module) -> dict[int, torch.Tensor]:
    """A copy of each parameter's gradient, scaled so the whole vector has length 1."""
    params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
    square = sum(float(p.grad.detach().float().pow(2).sum()) for p in params)
    scale = 0.0 if square < 1e-12 else square**-0.5
    return {id(p): p.grad.detach() * scale for p in params}


def fit_rank1(
    model,
    loop: list[Trace],
    clean: list[Trace],
    layers: list[int],
    u_init: dict[int, torch.Tensor],
    epochs: int = 8,
    lr_u: float = 1e-2,
    lr_v: float = 1e-3,
    seed: int = 0,
    hold_id: int | None = None,
) -> tuple[Rank1Write, list[dict]]:
    """Adam on u and v. Each step is one looping trace plus one clean trace. Returns the edit and the epoch means."""
    from .model import layers as blocks

    if not loop:
        raise ValueError("no looping traces to fit")
    intermediate = blocks(model)[layers[0]].mlp.down_proj.in_features
    edit = Rank1Write(layers, intermediate, u_init, seed).to(next(model.parameters()).device)
    opt = torch.optim.Adam([{"params": edit.u.parameters(), "lr": lr_u}, {"params": edit.v.parameters(), "lr": lr_v}])
    loop_crops = [c for t in loop if (c := pair_crop(t)) is not None]
    clean_crops = [c for t in clean if (c := clean_crop(t)) is not None]
    if not loop_crops:
        raise ValueError("no looping trace had a usable window")
    history = []
    was_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    try:
        with edit.applied(model):
            for epoch in range(epochs):
                totals: dict[str, list[float]] = {}
                for i, crop in enumerate(loop_crops):
                    loss, parts = sequence_loss(model, crop, hold_id)
                    if hold_id is None:
                        loss.backward()
                    else:
                        # A summed backward spends the clipped step on the repeat. Split the step
                        # so </think> cannot hide inside that term, and match the clean trace's length.
                        _backward_equal(edit, list(parts.values()))
                    if epoch == 0 and i == 0:
                        grad = edit.v[str(layers[0])].grad
                        if grad is None or float(grad.abs().sum()) == 0:
                            raise RuntimeError("the rank-1 gains got no gradient; the backward did not reach the edit")
                    stats = {k: float(v.detach()) for k, v in parts.items()}
                    if hold_id is not None:
                        acc = _unit_grad(edit)
                        edit.zero_grad()
                    if clean_crops:
                        clean_loss, clean_parts = sequence_loss(model, clean_crops[i % len(clean_crops)])
                        clean_loss.backward()
                        stats["clean"] = float(clean_parts["keep"].detach())
                    if hold_id is not None:
                        clean_grad = _unit_grad(edit)
                        acc = {k: acc.get(k, 0) + clean_grad.get(k, 0) for k in set(acc) | set(clean_grad)}
                        for p in edit.parameters():
                            if p.requires_grad:
                                p.grad = acc.get(id(p))
                    torch.nn.utils.clip_grad_norm_(edit.parameters(), 1.0)
                    opt.step()
                    for k, v in stats.items():
                        totals.setdefault(k, []).append(v)
                row = {k: round(sum(v) / len(v), 5) for k, v in totals.items()} | {"epoch": epoch}
                history.append(row)
                from .cli import log

                log(f"epoch {epoch}: {row}")
    finally:
        model.config.use_cache = was_cache
    return edit, history


@torch.no_grad()
def continuation_batch(model, tok, rows: list[tuple[list[int], list[int]]], max_new: int, seed: int) -> list[list[int]]:
    """Continue each (prompt, response prefix). The presence penalty sees the response prefix, and </think> stops."""
    from .generate import Sampling, sample, stop_ids

    dev = next(model.parameters()).device
    n = len(rows)
    width = max(len(p) + len(r) for p, r in rows)
    ids = torch.zeros(n, width, dtype=torch.long, device=dev)
    mask = torch.zeros_like(ids)
    for i, (prompt, prefix) in enumerate(rows):
        x = prompt + prefix
        ids[i, width - len(x) :] = torch.tensor(x, device=dev)
        mask[i, width - len(x) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, logits, pos = out.past_key_values, out.logits[:, -1], pos[:, -1:] + 1
    seen = torch.zeros(n, logits.shape[-1], device=dev)
    for i, (_, prefix) in enumerate(rows):
        if prefix:
            seen[i, torch.tensor(prefix, device=dev)] = 1.0
    sampling = Sampling(max_new_tokens=max_new)
    gen = torch.Generator(device=dev).manual_seed(seed)
    think_id, stops = tok.convert_tokens_to_ids(THINK_END), stop_ids(model, tok)
    toks: list[list[int]] = [[] for _ in range(n)]
    done = [False] * n
    active = list(range(n))
    while True:
        nxt = sample(logits, seen, sampling, gen).tolist()
        for j, r in enumerate(active):
            toks[r].append(nxt[j])
            if nxt[j] in stops or nxt[j] == think_id or len(toks[r]) >= max_new:
                done[r] = True
        step = torch.tensor(nxt, device=dev)
        seen[torch.arange(len(active), device=dev), step] = 1.0
        keep = [j for j, r in enumerate(active) if not done[r]]
        if not keep:
            return toks
        if len(keep) < len(active):
            k = torch.tensor(keep, device=dev)
            cache.reorder_cache(k)
            mask, pos, seen, step = mask[k], pos[k], seen[k], step[k]
            active = [active[j] for j in keep]
        mask = torch.cat([mask, mask.new_ones(len(active), 1)], dim=1)
        out = model(
            input_ids=step[:, None], attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True
        )
        logits, pos = out.logits[:, -1], pos + 1


def load_manifest(path: str | None = None) -> dict:
    return json.loads((Path(path) if path else ROOT / "configs" / "branch_val.json").read_text())


def branch_summary(rows: list[dict], offsets: list[int]) -> dict:
    out = {}
    for off in offsets:
        group = [r for r in rows if r["offset"] == off]
        n = len(group)
        out[str(off)] = {
            "n": n,
            "closed": round(sum(r["closed"] for r in group) / n, 4),
            "looping": round(sum(r["looping"] for r in group) / n, 4),
        }
    return out


def _select(traces: list[Trace], split: str, heldout: set[str], kind: str, limit: int) -> list[Trace]:
    if kind == "loop":
        ok = [t for t in traces if split_of(t.id) == split and t.id not in heldout and pair_crop(t)]
        return sorted(ok, key=lambda t: (t.onset, t.id))[:limit]
    ok = [t for t in traces if split_of(t.id) == split and t.id not in heldout and clean_crop(t)]
    return sorted(ok, key=lambda t: t.id)[:limit]


@torch.no_grad()
def _mean_loss(model, crops: list[dict], hold_id: int | None = None) -> dict:
    totals: dict[str, list[float]] = {}
    for crop in crops:
        _, parts = sequence_loss(model, crop, hold_id)
        for k, v in parts.items():
            totals.setdefault(k, []).append(float(v.detach()))
    return {k: round(sum(v) / len(v), 5) for k, v in totals.items()}


def cmd_fit_rank1(args) -> None:
    from .cli import environment, log, write_json
    from .conditions import load_directions, save_directions
    from .model import load, writers

    model, tok = load("qwen3.5-0.8b")
    model.requires_grad_(False)
    manifest_ids = {t["id"] for t in load_manifest()["traces"]}
    traces = load_traces("qwen3.5-0.8b", tok, LoopRule())
    loop = _select(traces, "fit", manifest_ids, "loop", args.max_traces)
    clean = _select(traces, "fit", manifest_ids, "clean", args.max_traces // 2)
    val_loop = _select(traces, "val", manifest_ids, "loop", args.max_traces)
    val_clean = _select(traces, "val", manifest_ids, "clean", args.max_traces // 2)
    log(f"fit traces: {len(loop)} looping, {len(clean)} clean; val: {len(val_loop)} looping, {len(val_clean)} clean")
    dirs = load_directions()
    missing = [L for L in args.layers if f"diffmean.u.{L}" not in dirs]
    if missing:
        raise SystemExit(f"missing diffmean.u for layers {missing}; run `unloop directions` first")
    u_init = {L: dirs[f"diffmean.u.{L}"] for L in args.layers}
    hold_id = None
    if args.hold_think:
        hold_id = tok.convert_tokens_to_ids(THINK_END)
        if not isinstance(hold_id, int) or hold_id < 0:
            raise SystemExit(f"tokenizer has no id for {THINK_END}")
        log(f"holding {THINK_END} (id {hold_id}) on the loop stretch")
    name = "rank1_hold" if args.hold_think else "rank1"
    before = {
        "loop": _mean_loss(model, [pair_crop(t) for t in val_loop], hold_id),
        "clean": _mean_loss(model, [clean_crop(t) for t in val_clean]),
    }
    edit, history = fit_rank1(
        model,
        loop,
        clean,
        list(args.layers),
        u_init,
        args.epochs,
        args.lr_u,
        args.lr_v,
        args.seed,
        hold_id,
    )
    saved = {}
    geometry = []
    for L in args.layers:
        u, v = unit(edit.direction(L).detach().float().cpu()), edit.v[str(L)].detach().float().cpu()
        saved[f"{name}.u.{L}"] = u
        saved[f"{name}.v.{L}"] = v
        W = writers(model, L)["mlp"].weight.detach().float().cpu()
        mass = v.abs()
        row = {
            "layer": L,
            "v_norm": round(float(v.norm()), 5),
            "top64_l1_share": round(float(mass.topk(min(64, mass.numel())).values.sum() / mass.sum()), 4),
            "cos_diffmean": round(cosine(u, dirs[f"diffmean.u.{L}"]), 4),
            "cos_v_with_existing_write": round(cosine(v, u @ W), 4),
        }
        if f"released.u.{L}" in dirs:
            row["cos_released"] = round(cosine(u, dirs[f"released.u.{L}"]), 4)
        geometry.append(row)
        log(str(row))
    save_directions(saved)
    from contextlib import ExitStack

    from .conditions import apply_condition

    with ExitStack() as stack:
        apply_condition(model, {"edit": {"direction": name, "layers": list(args.layers)}}, stack)
        after = {
            "loop": _mean_loss(model, [pair_crop(t) for t in val_loop], hold_id),
            "clean": _mean_loss(model, [clean_crop(t) for t in val_clean]),
        }
    write_json(
        RESULTS / f"{name}.json",
        {
            "hold_think": bool(args.hold_think),
            "layers": list(args.layers),
            "epochs": args.epochs,
            "lr_u": args.lr_u,
            "lr_v": args.lr_v,
            "seed": args.seed,
            "context": CONTEXT,
            "break_len": BREAK_LEN,
            "max_len": MAX_LEN,
            "n_fit_loop": len(loop),
            "n_fit_clean": len(clean),
            "n_val_loop": len(val_loop),
            "n_val_clean": len(val_clean),
            "history": history,
            "val_loss_before": before,
            "val_loss_after": after,
            "geometry": geometry,
            "env": environment(),
        },
    )


def cmd_branch(args) -> None:
    """Restart generation from the pilot's prefixes and, on the same model, try the clean control set."""
    from contextlib import ExitStack

    from .cli import environment, log, write_json
    from .conditions import apply_condition
    from .config import load_run_config
    from .model import load

    cfg = load_run_config(args.config)
    cond = {} if args.condition == "base" else dict(cfg["conditions"][args.condition])
    model, tok = load(cond.pop("model", "qwen3.5-0.8b"))
    manifest = load_manifest(args.manifest)
    by_id = {t.id: t for t in load_traces("qwen3.5-0.8b", tok, LoopRule())}
    chosen = []
    for row in manifest["traces"]:
        trace = by_id[row["id"]]
        onset = int(row["onset"])
        if onset + max(manifest["offsets"]) > len(trace.response):
            raise ValueError(f"{row['id']}: onset {onset} leaves no room for the last offset")
        chosen.append((trace, onset))
    offsets = list(manifest["offsets"])
    jobs = [(i, off, k) for i in range(len(chosen)) for off in offsets for k in range(args.samples)]
    jobs.sort(key=lambda j: -(len(chosen[j[0]][0].prompt) + chosen[j[0]][1] + j[1]))
    rule, think_id, rows = LoopRule(), tok.convert_tokens_to_ids(THINK_END), []
    from .generate import stop_ids

    stops = stop_ids(model, tok)
    log(f"{args.condition}: {len(jobs)} restarts")
    with ExitStack() as stack:
        apply_condition(model, cond, stack)
        for b in range(0, len(jobs), args.batch_size):
            chunk = jobs[b : b + args.batch_size]
            batch = [(chosen[i][0].prompt, chosen[i][0].response[: chosen[i][1] + off]) for i, off, _ in chunk]
            for (i, off, k), (_prompt, prefix), new in zip(
                chunk, batch, continuation_batch(model, tok, batch, args.max_new, args.seed + b)
            ):
                rows.append(
                    {"trace": chosen[i][0].id, "onset": chosen[i][1], "offset": off, "sample": k}
                    | continuation_outcome(prefix, new, think_id, stops, rule)
                )
            log(f"batch {b // args.batch_size + 1}/{-(-len(jobs) // args.batch_size)}")
        summary = branch_summary(rows, offsets)
        log(json.dumps(summary))
        control = _control(model, tok, by_id, manifest, cfg, args)
    write_json(
        RESULTS / f"branch_{args.condition}.json",
        {
            "condition": args.condition,
            "spec": cond,
            "max_new": args.max_new,
            "samples": args.samples,
            "summary": summary,
            "control": control,
            "rows": rows,
            "env": environment(),
        },
    )


def _control(model, tok, by_id, manifest, cfg, args) -> dict:
    """Regenerate finished, non-looping val traces from the prompt, past </think>, and grade them."""
    from .cli import log
    from .data import chat_prompt, load_problems
    from .generate import Decoding, Sampling, generate_hf
    from .grading import extract_answer, is_correct

    held = {t["id"] for t in manifest["traces"]}
    clean = _select(list(by_id.values()), "val", held, "clean", 16)
    if not clean:
        return {"n": 0}
    problems = {p.id: p for ds in {t.dataset for t in clean} for p in load_problems(ds)}
    items = [(t, k) for t in clean for k in range(min(4, args.samples))]
    sampling = Sampling(**{**cfg["sampling"], "max_new_tokens": args.max_new, "seed": args.seed})
    prompts = [chat_prompt(tok, problems[t.id].problem) for t, _ in items]
    recs = generate_hf(model, tok, prompts, sampling, Decoding(), args.batch_size, log)
    rule = LoopRule()
    graded = []
    for (t, k), rec in zip(items, recs):
        text = tok.decode(rec["tokens"], skip_special_tokens=False)
        pred = extract_answer(text)
        graded.append(
            {
                "trace": t.id,
                "sample": k,
                "correct": is_correct(pred, problems[t.id].answer),
                "loop": loop_onset(rec["tokens"], rule) is not None,
                "think_closed": THINK_END in text,
                "n_tokens": len(rec["tokens"]),
            }
        )
    n = len(graded)
    return {
        "n": n,
        "correct": round(sum(r["correct"] for r in graded) / n, 4),
        "loop": round(sum(r["loop"] for r in graded) / n, 4),
        "think_closed": round(sum(r["think_closed"] for r in graded) / n, 4),
        "rows": graded,
    }


def add_commands(sub) -> None:
    p = sub.add_parser("fit-rank1", help="train the rank-1 escape edit on fit traces")
    p.add_argument("--layers", type=int, nargs="+", default=[15, 19])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-traces", type=int, default=64)
    p.add_argument("--lr-u", type=float, default=1e-2)
    p.add_argument("--lr-v", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-think", action="store_true", help="penalize </think> on the loop stretch")
    p.set_defaults(fn=cmd_fit_rank1)
    p = sub.add_parser("branch", help="escape rate from the pilot's restart prefixes")
    p.add_argument("--condition", required=True)
    p.add_argument("--config", default="configs/full.toml")
    p.add_argument("--manifest")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-new", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_branch)

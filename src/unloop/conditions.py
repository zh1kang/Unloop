"""Turning a condition spec into weight edits and forward hooks on a loaded model.

Spec keys (all optional): `model`, `edit`, `hook`, `sampling`, `decoding`.
  edit = {direction, layers, writers = ["mlp"], alpha = 1.0, top_k}   W <- W - alpha u (u^T W) on top_k columns
  a direction with a saved v is a trained edit: W <- W + u v^T, and top_k keeps the largest |v| entries
  hook = {kind = "steer" | "ablate", direction, layers, scale}         residual stream edits at inference time
`edit = "sweep"` or `hook = "sweep"` takes the spec that `unloop sweep` selected.
Directions are read from results/directions.safetensors: "<source>.u.<layer>" (unit, at the output of that
layer), "<source>.r.<layer>" (raw loop-minus-productive mean), "<source>.cols.<layer>", "<source>.alpha.<layer>".
"""

from __future__ import annotations

import json
from contextlib import ExitStack

from .config import RESULTS

DIRECTIONS = RESULTS / "directions.safetensors"
SWEEP = RESULTS / "sweep.json"


def load_directions() -> dict:
    from safetensors.torch import load_file

    return load_file(DIRECTIONS)


def save_directions(new: dict) -> None:
    from safetensors.torch import save_file

    old = load_directions() if DIRECTIONS.exists() else {}
    DIRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in (old | new).items()}, DIRECTIONS)


def resolve(cond: dict) -> dict:
    cond = dict(cond)
    for key in ("edit", "hook"):
        if cond.get(key) == "sweep":
            cond[key] = json.loads(SWEEP.read_text())["best"][key]
    return cond


def weight_edits(model, spec: dict, dirs: dict) -> dict:
    from .edits import keep_topk, project_out, rank1_update
    from .model import writers

    names = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for layer in spec["layers"]:
        src = spec["direction"]
        if f"{src}.v.{layer}" in dirs:  # trained u, v; not the project_out family
            u, v = dirs[f"{src}.u.{layer}"], dirs[f"{src}.v.{layer}"]
            if spec.get("top_k"):
                v = keep_topk(v, int(spec["top_k"]))
            W = writers(model, layer)["mlp"].weight
            out[names[id(W)]] = rank1_update(W.detach().float().cpu(), u, v)
            continue
        u = dirs[f"{src}.u.{layer}"]
        alpha = float(dirs[f"{src}.alpha.{layer}"]) if spec.get("alpha") == "fit" else spec.get("alpha", 1.0)
        cols = dirs[f"{src}.cols.{layer}"][: spec["top_k"]] if spec.get("top_k") else None
        for kind in spec.get("writers", ["mlp"]):
            W = writers(model, layer)[kind].weight
            out[names[id(W)]] = project_out(W.detach().float().cpu(), u, alpha, cols)
    return out


def apply_condition(model, cond: dict, stack: ExitStack) -> None:
    """Apply a condition's edits and hooks; closing `stack` removes the hooks and restores the weights."""
    from .model import ablate, apply_weight_edits, residual_hooks, steer

    cond = resolve(cond)
    if "edit" not in cond and "hook" not in cond:
        return
    dirs = load_directions()
    if "edit" in cond:
        old = apply_weight_edits(model, weight_edits(model, cond["edit"], dirs))
        stack.callback(apply_weight_edits, model, old)
    if "hook" in cond:
        h = cond["hook"]
        layers = range(len(model.model.layers)) if h["layers"] == "all" else h["layers"]
        for layer in layers:
            if h["kind"] == "steer":  # move by `scale` times the loop-minus-productive mean difference
                r = dirs[f"{h['direction']}.r.{layer}"]
                fn = steer(r / r.norm(), h["scale"] * float(r.norm()))
            else:
                fn = ablate(dirs[f"{h['direction']}.u.{layer}"])
            stack.enter_context(residual_hooks(model, fn, at=[layer]))

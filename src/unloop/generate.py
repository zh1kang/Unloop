"""Batched sampling with the blog's settings, plus decoding-time interventions.

Two backends: `generate_hf` (any device; supports forward hooks and every decoding option) and `generate_vllm`
(CUDA; weight edits through a saved checkpoint, presence penalty and the thinking budget).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import THINK_END

# Thinking-budget text from the Qwen3 technical report (section 4.3), followed by the end-of-thinking token.
BUDGET_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
    + THINK_END
    + "\n\n"
)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    max_new_tokens: int = 32768
    seed: int = 0


@dataclass(frozen=True)
class Decoding:
    think_bias: float = 0.0  # added to the </think> logit ...
    think_bias_after: int = 0  # ... from this many generated tokens on, while thinking is open
    think_budget: int | None = None  # force BUDGET_TEXT after this many tokens if thinking is still open


def stop_ids(model, tok) -> set[int]:
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    return {i for i in ids if i is not None}


def sample(logits, seen, s: Sampling, generator, bias_rows=None, think_id=None, think_bias=0.0):
    """One sampling step in vLLM's order: penalties, biases, temperature, top-k, top-p."""
    logits = logits.float() - s.presence_penalty * seen
    if bias_rows is not None and think_bias:
        logits[:, think_id] += think_bias * bias_rows
    if s.temperature == 0:
        return logits.argmax(-1)
    vals, ids = (logits / s.temperature).topk(s.top_k if s.top_k > 0 else logits.shape[-1])
    probs = vals.softmax(-1)
    probs = probs.masked_fill(probs.cumsum(-1) - probs > s.top_p, 0.0)
    return ids.gather(-1, torch.multinomial(probs, 1, generator=generator)).squeeze(-1)


@torch.no_grad()
def generate_hf(model, tok, prompts: list[str], s: Sampling, d: Decoding = Decoding(), batch_size: int = 32, log=None):
    """Returns one {"tokens", "finish", "forced"} record per prompt. Batches are sorted by prompt length."""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[dict] = [{}] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        for i, rec in zip(idx, _decode_batch(model, tok, [prompts[i] for i in idx], s, d, s.seed + b)):
            out[i] = rec
        if log:
            log(f"batch {b // batch_size + 1}/{-(-len(order) // batch_size)} done")
    return out


def _decode_batch(model, tok, prompts, s: Sampling, d: Decoding, seed: int) -> list[dict]:
    dev = model.device
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(dev)
    mask = enc.attention_mask
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=enc.input_ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, logits, pos = out.past_key_values, out.logits[:, -1], pos[:, -1:] + 1
    eos, think_id = stop_ids(model, tok), tok.convert_tokens_to_ids(THINK_END)
    budget_ids = tok(BUDGET_TEXT, add_special_tokens=False).input_ids if d.think_budget is not None else []
    gen = torch.Generator(device=dev).manual_seed(seed)

    n = len(prompts)
    tokens: list[list[int]] = [[] for _ in range(n)]
    queue: list[list[int]] = [[] for _ in range(n)]
    closed, forced, finish = [False] * n, [False] * n, [""] * n
    seen = torch.zeros(n, logits.shape[-1], device=dev)
    active = list(range(n))  # original row of each cache row
    while True:
        bias_rows = torch.tensor(
            [float(not closed[r] and len(tokens[r]) >= d.think_bias_after) for r in active], device=dev
        )
        nxt = sample(logits, seen, s, gen, bias_rows, think_id, d.think_bias).tolist()
        for j, r in enumerate(active):
            if d.think_budget is not None and not closed[r] and not queue[r] and len(tokens[r]) >= d.think_budget:
                queue[r], forced[r] = list(budget_ids), True
            if queue[r]:
                nxt[j] = queue[r].pop(0)
            tokens[r].append(nxt[j])
            closed[r] |= nxt[j] == think_id
            if nxt[j] in eos:
                finish[r] = "stop"
            elif len(tokens[r]) >= s.max_new_tokens:
                finish[r] = "length"
        step = torch.tensor(nxt, device=dev)
        seen[torch.arange(len(active), device=dev), step] = 1.0
        keep = [j for j, r in enumerate(active) if not finish[r]]
        if not keep:
            break
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
    return [{"tokens": tokens[r], "finish": finish[r], "forced": forced[r]} for r in range(n)]


def generate_vllm(model_path: str, tok, prompts: list[str], s: Sampling, d: Decoding = Decoding()) -> list[dict]:
    """vLLM generation for weight-edited checkpoints. The thinking budget runs as a second pass."""
    from vllm import LLM, SamplingParams

    if d.think_bias:
        raise NotImplementedError("think_bias needs a logits processor; run this condition with --backend hf")
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=s.max_new_tokens + 2048, seed=s.seed)

    def params(max_tokens: int, rows: list[int]) -> list[SamplingParams]:
        # one seed per request, so that repeated prompts give independent samples
        return [
            SamplingParams(
                temperature=s.temperature,
                top_p=s.top_p,
                top_k=s.top_k,
                presence_penalty=s.presence_penalty,
                max_tokens=max_tokens,
                seed=s.seed + i,
            )
            for i in rows
        ]

    first = llm.generate(prompts, params(d.think_budget or s.max_new_tokens, list(range(len(prompts)))))
    recs = [
        {"tokens": list(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason, "forced": False} for o in first
    ]
    if d.think_budget is None:
        return recs
    think_id = tok.convert_tokens_to_ids(THINK_END)
    todo = [i for i, r in enumerate(recs) if r["finish"] == "length"]
    for i in todo:
        if think_id not in recs[i]["tokens"]:
            recs[i]["tokens"] += tok(BUDGET_TEXT, add_special_tokens=False).input_ids
            recs[i]["forced"] = True
    conts = [tok.decode(recs[i]["tokens"]) for i in todo]
    rest = s.max_new_tokens - d.think_budget
    second = llm.generate([prompts[i] + c for i, c in zip(todo, conts)], params(rest, todo)) if todo else []
    for i, o in zip(todo, second):
        recs[i]["tokens"] += list(o.outputs[0].token_ids)
        recs[i]["finish"] = o.outputs[0].finish_reason
    return recs

"""Optional contrast: the loop rate of a hosted model through an OpenAI-compatible chat API.

A hosted API gives no weights or activations, so only the loop rate and accuracy are measured. The key is read
from a file and sent only in the Authorization header. Loops are detected on the Qwen3.5 tokenization of the
returned text, which approximates the server's own tokens.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def chat(base_url: str, key: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def cmd_api_loops(args) -> None:
    from .cli import problems_for, summarize, write_json
    from .config import INSTRUCTION, MODELS, RESULTS, load_run_config
    from .grading import extract_answer, is_correct
    from .loops import LoopRule, loop_onset

    cfg = load_run_config(args.config)
    s, rule = cfg["sampling"], LoopRule(**cfg.get("loop", {}))
    key = Path(args.key_file).read_text().strip()
    max_tokens = args.max_tokens or s["max_new_tokens"]
    items, count = [], {}
    for p, k in problems_for(cfg):  # the first n problems of each benchmark's subset, one sample each
        if k == 0 and count.get(p.dataset, 0) < args.n:
            items.append((p, k))
            count[p.dataset] = count.get(p.dataset, 0) + 1

    def run(item):
        p, _ = item
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": p.problem + "\n" + INSTRUCTION}],
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "presence_penalty": s["presence_penalty"],
            "max_tokens": max_tokens,
            # accepted by vLLM and SGLang servers; other servers may ignore or reject them
            "top_k": s["top_k"],
            "chat_template_kwargs": {"enable_thinking": True},
        }
        choice = chat(args.base_url, key, body)["choices"][0]
        msg = choice["message"]
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return thinking + ("</think>\n\n" if thinking else "") + (msg.get("content") or ""), choice["finish_reason"]

    from transformers import AutoTokenizer

    repo, rev = MODELS["qwen3.5-0.8b"]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    with ThreadPoolExecutor(args.workers) as pool:
        outs = list(pool.map(run, items))
    rows = []
    for (p, k), (text, finish) in zip(items, outs):
        ids = tok(text, add_special_tokens=False).input_ids
        pred = extract_answer(text)
        rows.append(
            {
                "id": p.id,
                "dataset": p.dataset,
                "sample": k,
                "n_tokens": len(ids),
                "finish": "length" if finish == "length" else "stop",
                "think_closed": "</think>" in text,
                "pred": pred,
                "correct": is_correct(pred, p.answer),
                "loop": loop_onset(ids, rule) is not None,
            }
        )
    write_json(
        RESULTS / "api" / f"{args.model.replace('/', '_')}.json",
        {"model": args.model, "sampling": s, "max_tokens": max_tokens, "metrics": summarize(rows), "per_sample": rows},
    )


def add_commands(sub) -> None:
    p = sub.add_parser("api-loops", help="loop rate of a hosted model (OpenAI-compatible API)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file", required=True, help="file that holds the API key")
    p.add_argument("--config", default="configs/full.toml")
    p.add_argument("--n", type=int, default=20, help="problems per benchmark")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_api_loops)

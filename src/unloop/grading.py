"""Answer extraction and checking, matching the blog's protocol (exp8_math/code/mathqa.py).

The answer is the content of the last \\boxed{} (or \\fbox{}) after the last </think>; with no </think>
the whole response is read. Equivalence is decided by math-verify 0.9.0.
"""

from __future__ import annotations

import threading
from functools import lru_cache


def answer_text(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response


def _box_at(text: str, idx: int) -> str | None:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    stripped = rest.lstrip()
    if stripped.startswith("{"):
        depth = 0
        for k, ch in enumerate(stripped):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return stripped[1:k].strip() or None
        return None  # unclosed box, e.g. a response cut inside it
    if key == "\\boxed" and rest.startswith(" "):  # "\boxed 5": read up to "$" or the line end
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    return None if idx < 0 else _box_at(text, idx)


def extract_answer(response: str) -> str | None:
    return last_boxed(answer_text(response))


@lru_cache(maxsize=1 << 14)
def _parse(answer: str):
    from math_verify import parse

    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    return parse("\\boxed{" + answer + "}", parsing_timeout=timeout)


def is_correct(pred: str | None, gold: str | None) -> bool:
    if not pred or not gold:
        return False
    from math_verify import verify

    g, p = _parse(gold), _parse(pred)
    if not g or not p:
        return False
    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    try:
        return bool(verify(g, p, timeout_seconds=timeout))
    except Exception:  # math-verify raises on some malformed inputs; the blog counts these as wrong
        return False

"""Loop detection on token ids.

A token is "repeated" when the n-gram that ends at it already ended at an earlier position.
A response is looping when some suffix of at least `min_len` tokens is at least `tau` repeated and contains a
`min_len`-token window that is itself that repeated:
the model is stuck in a verbatim (or near-verbatim) cycle at the point where it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopRule:
    n: int = 16  # n-gram size in tokens
    min_len: int = 256  # shortest suffix that can count as a loop
    tau: float = 0.9  # share of repeated tokens in that suffix
    stop_window: int = 1024  # online early stop: the last `stop_window` tokens ...
    stop_tau: float = 0.98  # ... are at least this repeated


def repeated_mask(tokens: list[int], n: int) -> list[bool]:
    seen: set[tuple[int, ...]] = set()
    out = [False] * len(tokens)
    for i in range(n - 1, len(tokens)):
        gram = tuple(tokens[i - n + 1 : i + 1])
        if gram in seen:
            out[i] = True
        else:
            seen.add(gram)
    return out


def loop_onset(tokens: list[int], rule: LoopRule = LoopRule()) -> int | None:
    """Where the final loop starts, or None if the response is not looping.

    A response is looping when its longest suffix that is at least `tau` repeated has `min_len` tokens or more.
    Over a long loop, that suffix's 10% slack also takes in thousands of earlier, sparsely repeated tokens, so
    the onset is the first repeated token of the first `min_len`-token window inside the suffix that is itself
    at least `tau` repeated: the start of the dense repetition.
    """
    mask = repeated_mask(tokens, rule.n)
    suffix, best = 0, None
    for s in range(len(tokens) - 1, -1, -1):
        suffix += mask[s]
        if len(tokens) - s >= rule.min_len and suffix >= rule.tau * (len(tokens) - s):
            best = s
    if best is None:
        return None
    w = rule.min_len
    run = sum(mask[best : best + w])
    for s in range(best, len(tokens) - w + 1):
        if run >= rule.tau * w:
            return mask.index(True, s)
        if s + w < len(tokens):
            run += mask[s + w] - mask[s]
    return None  # no dense window: sparse repeats only, not a loop


def first_stuck(tokens: list[int], rule: LoopRule = LoopRule()) -> int | None:
    """End of the first window of `stop_window` tokens that is at least `stop_tau` repeated, or None.

    An online early-stop rule would cut generation here; `reproduce` measures how often such traces still finish.
    """
    mask = repeated_mask(tokens, rule.n)
    w = rule.stop_window
    run = sum(mask[:w])
    for end in range(w, len(tokens) + 1):
        if end >= w + rule.n and run >= rule.stop_tau * w:
            return end
        if end < len(tokens):
            run += mask[end] - mask[end - w]
    return None

"""Direction math and weight edits (pure tensor functions; no model needed)."""

from __future__ import annotations

import torch


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(unit(a.flatten().double()) @ unit(b.flatten().double()))


def changed(base: torch.Tensor, edited: torch.Tensor) -> dict:
    """Which entries and columns of a weight differ, compared bit for bit."""
    diff = base != edited
    return {
        "shape": list(base.shape),
        "changed_entries": int(diff.sum()),
        "changed_columns": int(diff.any(0).sum()),
        "changed_rows": int(diff.any(1).sum()),
        "columns": diff.any(0).nonzero().flatten().tolist(),
    }


def rank1(delta: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """(u, sigma_1, sigma_2 / sigma_1) of a weight delta; u is the residual-stream direction it writes."""
    U, S, _ = torch.linalg.svd(delta.double(), full_matrices=False)
    return U[:, 0].float(), float(S[0]), float(S[1] / S[0])


def project_out(W: torch.Tensor, u: torch.Tensor, alpha: float = 1.0, cols=None) -> torch.Tensor:
    """W - alpha * u (u^T W), on the given columns only (all if None).

    alpha = 1 removes the u component a writer adds to the residual stream (directional ablation, Arditi et al.);
    alpha > 1 reverses it. The released edit has this form with alpha = 6 on 64 columns.
    """
    Wf, u = W.double(), unit(u.double())
    coef = u @ Wf
    if cols is not None:
        keep = torch.zeros_like(coef, dtype=torch.bool)
        keep[torch.as_tensor(cols)] = True
        coef = coef * keep
    return (Wf - alpha * torch.outer(u, coef)).to(W.dtype)


def fit_alpha(W0: torch.Tensor, W1: torch.Tensor, u: torch.Tensor, cols) -> float:
    """The alpha of project_out that best maps W0 to W1 on `cols` (least squares)."""
    u, cols = unit(u.double()), torch.as_tensor(cols)
    a = u @ W0.double()[:, cols]
    b = u @ W1.double()[:, cols]
    return float(((a - b) @ a) / (a @ a))


def diff_means(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """mean(pos) - mean(neg) over the first axis, for [n, d] (or [n, layers, d]) inputs."""
    return pos.double().mean(0) - neg.double().mean(0)


def top_writers(act_diff: torch.Tensor, W: torch.Tensor, u: torch.Tensor, k: int) -> list[int]:
    """The k MLP neurons whose extra activation in loops writes the most along u: (a_loop - a_prod)_j * u^T W[:, j]."""
    score = act_diff.double() * (unit(u.double()) @ W.double())
    return sorted(score.topk(min(k, score.numel())).indices.tolist())


def logit_lens(u: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Logit change per unit of u added before the final norm (ignoring the norm's rescaling)."""
    return W_U.double() @ (gain.double() * unit(u.double()))

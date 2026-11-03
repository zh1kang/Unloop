"""Loading pinned models and reading or editing the residual stream with forward hooks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from .config import MODELS


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


ATTN = "sdpa_grouped"


def grouped_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """transformers' "sdpa" attention, except that one-token steps with a padding mask do not copy key/value heads.

    With a mask, "sdpa" repeats each key/value head for all of its query heads (grouped-query attention), which in
    batched decoding at 32k tokens copies gigabytes per layer per token. For a single query position, the query
    heads that share a key/value head can instead be laid out as rows of one query: [B, H, 1, D] -> [B, H_kv, G, D].
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    groups = getattr(module, "num_key_value_groups", 1)
    if query.shape[2] != 1 or attention_mask is None or groups == 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
    b, h, _, d = query.shape  # query head i uses key/value head i // groups, as in transformers' repeat_kv
    out = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(b, h // groups, groups, d), key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling
    )
    return out.reshape(b, 1, h, d), None  # [B, q_len, H, D], the layout transformers expects


def register_attention() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(ATTN, grouped_decode_attention)
    AttentionMaskInterface.register(ATTN, sdpa_mask)


def load(name: str, dev: str | None = None, dtype: torch.dtype = torch.bfloat16):
    """The text model (Qwen3_5ForCausalLM) and tokenizer of a pinned checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    register_attention()
    repo, rev = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype, attn_implementation=ATTN)
    return model.to(dev or device()).eval(), tok


def layers(model) -> nn.ModuleList:
    return model.model.layers


def writers(model, layer: int) -> dict[str, nn.Linear]:
    """The two matrices of a decoder layer that write into the residual stream."""
    block = layers(model)[layer]
    attn = block.linear_attn.out_proj if hasattr(block, "linear_attn") else block.self_attn.o_proj
    return {"attn": attn, "mlp": block.mlp.down_proj}


def unembed(model) -> tuple[torch.Tensor, torch.Tensor]:
    """(W_U [vocab, d], final norm gain [d]); Qwen3_5RMSNorm scales by (1 + weight)."""
    return model.lm_head.weight, 1.0 + model.model.norm.weight.float()


Hook = Callable[[int, torch.Tensor], torch.Tensor | None]


@contextmanager
def residual_hooks(model, fn: Hook, at: list[int] | None = None) -> Iterator[None]:
    """Call fn(layer, h) on the residual stream after each decoder layer in `at` (all if None).

    The output of layer i is hidden_states[i + 1] in Hugging Face's indexing. If fn returns a tensor it replaces h.
    """

    def make(i: int):
        def hook(_module, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            new = fn(i, h)
            if new is None:
                return None
            return (new, *out[1:]) if isinstance(out, tuple) else new

        return hook

    idx = range(len(layers(model))) if at is None else at
    handles = [layers(model)[i].register_forward_hook(make(i)) for i in idx]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def steer(direction: torch.Tensor, scale: float) -> Hook:
    """Activation addition: h <- h + scale * direction (direction is unit norm)."""
    return lambda _i, h: h + scale * direction.to(h)


def ablate(direction: torch.Tensor) -> Hook:
    """Directional ablation: h <- h - (h . u) u."""

    def fn(_i, h):
        u = direction.to(h)
        return h - (h @ u)[..., None] * u

    return fn


@torch.no_grad()
def apply_weight_edits(model, edits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace named parameters in place; returns the originals so the caller can restore them."""
    params = dict(model.named_parameters())
    old = {}
    for name, value in edits.items():
        p = params[name]
        if p.shape != value.shape:
            raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(p.shape)}")
        old[name] = p.detach().clone()
        p.copy_(value.to(p))
    return old

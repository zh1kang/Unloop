import sys

import pytest
import torch

# The tiny model runs on the CPU. When these CUDA/Triton kernel packages are installed (the cluster environment),
# transformers 5.20 calls them even for CPU tensors and fails; hiding them selects its PyTorch reference code.
for _kernel_pkg in ("causal_conv1d", "fla"):
    sys.modules[_kernel_pkg] = None


class IntTokenizer:
    """Whitespace-separated integer ids; enough of the tokenizer interface for the decode loop."""

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, texts, return_tensors=None, padding=False, padding_side="left", add_special_tokens=True):
        if isinstance(texts, str):
            return type("Enc", (), {"input_ids": [int(t) for t in texts.split()]})()
        rows = [[int(t) for t in s.split()] for s in texts]
        width = max(map(len, rows))
        ids = torch.tensor([[0] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        return _Enc(ids, mask)

    def convert_tokens_to_ids(self, token):
        return {"</think>": 2}[token]

    def decode(self, ids):
        return " ".join(map(str, ids))


class _Enc:
    def __init__(self, ids, mask):
        self.input_ids, self.attention_mask = ids, mask

    def to(self, dev):
        return _Enc(self.input_ids.to(dev), self.attention_mask.to(dev))


@pytest.fixture(scope="session")
def tiny():
    """A random 4-layer Qwen3.5 text model (3 Gated DeltaNet layers + 1 full-attention layer)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        eos_token_id=1,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    from unloop.model import ATTN, register_attention

    register_attention()
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.set_attn_implementation(ATTN)  # what `unloop.model.load` uses
    model.generation_config.eos_token_id = 1
    return model, IntTokenizer(64)

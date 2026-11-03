from contextlib import ExitStack

import pytest
import torch

from unloop.edits import changed, cosine, diff_means, fit_alpha, project_out, rank1, top_writers, unit
from unloop.model import ablate, apply_weight_edits, residual_hooks, steer, writers


@pytest.fixture
def released_like():
    """A synthetic edit of the released form: W1 = W0 - 6 u (mask * u^T W0) on 8 of 40 columns."""
    g = torch.Generator().manual_seed(0)
    W0 = torch.randn(16, 40, generator=g)
    u = unit(torch.randn(16, generator=g))
    cols = [1, 4, 9, 10, 22, 30, 31, 39]
    return W0, project_out(W0, u, alpha=6.0, cols=cols), u, cols


def test_project_out_alpha_one_removes_the_component():
    g = torch.Generator().manual_seed(1)
    W, u = torch.randn(8, 5, generator=g), unit(torch.randn(8, generator=g))
    assert torch.allclose(u @ project_out(W, u), torch.zeros(5), atol=1e-5)
    # columns outside `cols` are untouched
    out = project_out(W, u, cols=[0, 2])
    assert torch.equal(out[:, [1, 3, 4]], W[:, [1, 3, 4]])


def test_released_form_is_recovered(released_like):
    W0, W1, u, cols = released_like
    info = changed(W0, W1)
    assert info["changed_columns"] == len(cols) and info["columns"] == cols
    v, _, ratio = rank1(W1 - W0)
    assert abs(cosine(v, u)) > 0.9999 and ratio < 1e-5
    assert fit_alpha(W0, W1, v, cols) == pytest.approx(6.0, rel=1e-4)


def test_diff_means_and_top_writers():
    g = torch.Generator().manual_seed(2)
    d, n = 12, 30
    u = unit(torch.randn(d, generator=g))
    neg = torch.randn(200, d, generator=g)
    pos = torch.randn(200, d, generator=g) + 3 * u
    assert cosine(diff_means(pos, neg), u) > 0.95
    W = torch.randn(d, n, generator=g)
    W[:, 7] += 5 * u  # neuron 7 writes u and fires more in loops
    act = torch.zeros(n)
    act[7] = 1.0
    act[3] = 1.0  # neuron 3 fires more but does not write u in particular
    assert 7 in top_writers(act, W, u, 1)


def test_weight_edit_and_hooks_on_tiny_model(tiny):
    model, _ = tiny
    u = unit(torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(3)))
    ids = torch.tensor([[3, 9, 12, 5, 7]])
    down = writers(model, 1)["mlp"]
    name = next(n for n, p in model.named_parameters() if p is down.weight)

    old = apply_weight_edits(model, {name: project_out(down.weight.detach(), u)})
    try:
        captured = {}
        hook = lambda i, h: captured.setdefault(i, h)  # noqa: E731
        with residual_hooks(model, hook, at=[1]), torch.no_grad():
            model(input_ids=ids)
        # the edited MLP no longer writes u, so u is projected out of its output
        mlp_out = down(torch.randn(4, down.in_features))
        assert torch.allclose(mlp_out @ u, torch.zeros(4), atol=1e-5)
    finally:
        apply_weight_edits(model, old)
    assert torch.equal(down.weight, old[name])

    with ExitStack() as stack, torch.no_grad():
        stack.enter_context(residual_hooks(model, ablate(u), at=[2]))
        stack.enter_context(residual_hooks(model, lambda i, h: captured.__setitem__("ab", h), at=[2]))
        model(input_ids=ids)
    assert torch.allclose(captured["ab"] @ u, torch.zeros(1, 5), atol=1e-4)

    with torch.no_grad():
        with residual_hooks(model, lambda i, h: captured.__setitem__("plain", h), at=[2]):
            model(input_ids=ids)
        with ExitStack() as stack:
            stack.enter_context(residual_hooks(model, steer(u, 2.0), at=[2]))
            stack.enter_context(residual_hooks(model, lambda i, h: captured.__setitem__("st", h), at=[2]))
            model(input_ids=ids)
    assert torch.allclose(captured["st"] - captured["plain"], 2.0 * u.expand(1, 5, -1), atol=1e-4)


def test_grouped_decode_attention_matches_sdpa_with_repeated_heads():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from unloop.model import grouped_decode_attention

    torch.manual_seed(0)
    b, h_kv, groups, t, d = 3, 2, 4, 37, 16
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal = groups, True
    q = torch.randn(b, h_kv * groups, 1, d)
    k, v = torch.randn(b, h_kv, t, d), torch.randn(b, h_kv, t, d)
    mask = torch.ones(b, 1, 1, t, dtype=torch.bool)
    mask[0, ..., :5] = False  # left padding
    mask[2, ..., :11] = False
    want, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=0.25)
    got, _ = grouped_decode_attention(module, q, k, v, mask, scaling=0.25)
    assert got.shape == want.shape == (b, 1, h_kv * groups, d)
    assert torch.allclose(got, want, atol=1e-6)

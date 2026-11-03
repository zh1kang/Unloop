import random
from contextlib import ExitStack

import pytest
import torch

import unloop.analysis as A
import unloop.conditions as C


def looping_trace(seed: int, onset: int = 400) -> A.Trace:
    rng = random.Random(seed)
    prompt = [rng.randrange(3, 64) for _ in range(10)]
    cycle = [rng.randrange(3, 64) for _ in range(9)]
    resp = [rng.randrange(3, 64) for _ in range(onset)] + cycle * 50
    return A.Trace(f"t{seed}", "gsm8k", prompt, resp, onset, False)


@pytest.fixture
def directions_file(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DIRECTIONS", tmp_path / "directions.safetensors")
    return C.DIRECTIONS


def test_windows_are_on_either_side_of_the_onset():
    t = looping_trace(0)
    w = t.windows()
    assert len(w["prod"]) == len(w["loop"]) == A.WIN
    assert max(w["prod"]) < len(t.prompt) + t.onset <= min(w["loop"])


def test_capture_matches_hidden_states(tiny):
    model, _ = tiny
    t = looping_trace(1)
    ids = t.ids(t.windows()["loop"][-1] + 1)
    got = A.capture(model, ids, t.windows())
    with torch.no_grad():
        hs = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    pos = torch.tensor(t.windows()["loop"])
    for layer in range(len(model.model.layers) - 1):  # Hugging Face's last entry is after the final norm
        assert torch.allclose(got["loop"][layer], hs[layer + 1][0, pos].mean(0), atol=1e-5)
    assert got["mlp_loop"].shape == (len(model.model.layers), model.config.intermediate_size)


def test_fit_directions_and_conditions_round_trip(tiny, directions_file):
    model, _ = tiny
    traces = [looping_trace(s) for s in range(4)]
    dirs, stats = A.fit_directions(model, traces)
    assert stats["n_traces"] == 4 and len(stats["layers"]) == len(model.model.layers)
    assert torch.allclose(dirs["diffmean.u.2"].norm(), torch.tensor(1.0))
    C.save_directions(dirs)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    think = 2
    base = A.next_token_stats(model, traces[0], think)
    for spec in (
        {"edit": {"direction": "diffmean", "layers": [1, 2], "writers": ["mlp", "attn"]}},
        {"edit": {"direction": "diffmean", "layers": [1], "alpha": 6.0, "top_k": 5}},
        {"hook": {"kind": "steer", "direction": "diffmean", "layers": [2], "scale": -2.0}},
        {"hook": {"kind": "ablate", "direction": "diffmean", "layers": "all"}},
    ):
        with ExitStack() as stack:
            C.apply_condition(model, spec, stack)
            edited = A.next_token_stats(model, traces[0], think)
        assert edited != base
    for n, p in model.named_parameters():  # every edit was undone
        assert torch.equal(p, before[n]), n
    assert A.next_token_stats(model, traces[0], think) == base


def test_select_respects_the_productive_budget():
    spec = lambda k: {"edit": {"layers": [k]}}  # noqa: E731
    results = {
        "ablate_mlp_1": {"d_prod_logp": -0.5, "d_loop_logp": -3.0, "spec": spec(1)},  # damages productive text
        "ablate_mlp_2": {"d_prod_logp": -0.01, "d_loop_logp": -1.0, "spec": spec(2)},
        "ablate_mlp_3": {"d_prod_logp": 0.0, "d_loop_logp": -0.2, "spec": spec(3)},
        "steer_7_-1": {"d_prod_logp": 0.0, "d_loop_logp": -0.4, "spec": {"hook": {"layers": [7]}}},
    }
    best = A.select(results, max_prod_drop=0.02)
    assert best["edit_name"] == "ablate_mlp_2" and best["hook_name"] == "steer_7_-1"


def test_split_is_stable_and_balanced():
    ids = [f"gsm8k_test_{i}" for i in range(400)]
    fit = [i for i in ids if A.split_of(i) == "fit"]
    assert 150 < len(fit) < 250 and A.split_of(ids[0]) == A.split_of(ids[0])

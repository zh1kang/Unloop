import random
from contextlib import ExitStack

import torch

import unloop.analysis as A
import unloop.conditions as C
from unloop.edits import keep_topk, rank1_update, unit
from unloop.loops import LoopRule
from unloop.model import writers
from unloop.rank1 import (
    Rank1Write,
    branch_summary,
    clean_crop,
    continuation_outcome,
    escape_pair_loss,
    load_manifest,
    logp_at,
    pair_crop,
    sequence_loss,
)


def looping(seed: int, onset: int = 400) -> A.Trace:
    rng = random.Random(seed)
    prompt = [rng.randrange(3, 64) for _ in range(10)]
    cycle = [rng.randrange(3, 64) for _ in range(9)]
    resp = [rng.randrange(3, 64) for _ in range(onset)] + cycle * 50
    return A.Trace(f"t{seed}", "gsm8k", prompt, resp, onset, False)


def test_escape_pair_loss_penalizes_a_confident_repeat_and_not_a_kept_token():
    # two positions, vocab 3. Keep is sure and correct; the repeat is sure and wrong for unlikelihood.
    logp = torch.log(torch.tensor([[0.05, 0.05, 0.9], [0.02, 0.02, 0.96]]))
    targets = torch.tensor([2, 2])
    keep_only = escape_pair_loss(logp[:1], targets[:1], torch.tensor([False]))
    repeat = escape_pair_loss(logp[1:], targets[1:], torch.tensor([True]))
    unsure = escape_pair_loss(torch.log(torch.tensor([[0.4, 0.4, 0.2]])), targets[1:], torch.tensor([True]))
    assert keep_only < 0.2  # -log 0.9
    assert repeat > unsure > 0.2  # a confident repeat is a worse unlikelihood


def test_holding_a_token_costs_more_when_that_token_is_likely():
    targets = torch.tensor([2])
    mask = torch.tensor([True])
    # Same probability on the repeated token. Only the held token's probability changes.
    high = torch.log(torch.tensor([[0.50, 0.05, 0.45]]))
    low = torch.log(torch.tensor([[0.05, 0.50, 0.45]]))
    assert escape_pair_loss(high, targets, mask, hold_id=0) > escape_pair_loss(low, targets, mask, hold_id=0)
    assert torch.allclose(escape_pair_loss(high, targets, mask), escape_pair_loss(low, targets, mask))


def test_pair_crop_labels_productive_tokens_keep_and_repeated_loop_tokens_break():
    trace = looping(0)
    crop = pair_crop(trace)
    full = trace.prompt + trace.response
    assert crop["targets"] == [full[crop["start"] + i + 1] for i in crop["index"]]
    flags = crop["break"]
    assert flags[: A.WIN] == [False] * A.WIN and any(flags) and not all(flags)
    assert min(i for i, f in zip(crop["index"], flags) if f) > max(i for i, f in zip(crop["index"], flags) if not f)


def test_clean_crop_is_all_keep():
    trace = A.Trace("c", "gsm8k", [1, 2, 3], list(range(3, 300)), None, True)
    crop = clean_crop(trace)
    assert crop is not None and not any(crop["break"]) and len(crop["targets"]) == A.WIN


def test_continuation_outcome_matches_the_pilot_rule():
    rule = LoopRule(n=4, min_len=40, tau=0.9)
    prefix = [1, 2, 3, 4] * 30
    cycle = [7, 8, 9] * 40
    closed = continuation_outcome(prefix, [5, 2], think_id=2, stop={1}, rule=rule)
    assert closed["closed"] and not closed["looping"]
    looping = continuation_outcome(prefix, cycle, think_id=2, stop={1}, rule=rule)
    assert not looping["closed"] and looping["looping"]
    left = continuation_outcome(prefix, list(range(200, 500)), think_id=2, stop={1}, rule=rule)
    assert not left["closed"] and not left["looping"]


def test_keep_topk_and_rank1_update():
    v = torch.tensor([0.1, -3.0, 0.2, 4.0])
    kept = keep_topk(v, 2)
    assert torch.equal(kept, torch.tensor([0.0, -3.0, 0.0, 4.0]))
    W, u = torch.randn(6, 4), torch.randn(6)
    got = rank1_update(W, u, v)
    assert torch.allclose(got.double() - W.double(), torch.outer(unit(u.double()), v.double()))


def test_logp_at_is_the_full_distribution_at_the_scored_positions(tiny):
    model, _ = tiny
    crop = pair_crop(looping(1))
    got = logp_at(model, crop)
    ids = torch.tensor([crop["ids"]])
    pos = torch.arange(crop["start"], crop["start"] + len(crop["ids"])).unsqueeze(0)
    full = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0].float().log_softmax(-1)
    assert torch.allclose(got, full[crop["index"]], atol=1e-5)


def test_hook_matches_the_weight_update_and_sends_a_gradient(tiny):
    model, _ = tiny
    model.requires_grad_(False)
    layer, d, h = 1, model.config.hidden_size, model.config.intermediate_size
    u = unit(torch.randn(d))
    edit = Rank1Write([layer], h, {layer: u}, seed=0)
    crop = pair_crop(looping(2))
    with torch.no_grad(), edit.applied(model):
        hooked = logp_at(model, crop)
    W = writers(model, layer)["mlp"].weight
    old = W.detach().clone()
    W.copy_(rank1_update(old, edit.direction(layer).detach(), edit.v[str(layer)].detach()))
    try:
        with torch.no_grad():
            edited = logp_at(model, crop)
    finally:
        W.copy_(old)
    assert torch.allclose(hooked, edited, atol=1e-5)

    with edit.applied(model):
        loss, stats = sequence_loss(model, crop)
        assert set(stats) == {"keep", "break"}
        loss.backward()
    assert edit.v[str(layer)].grad is not None and float(edit.v[str(layer)].grad.abs().sum()) > 0
    assert edit.u[str(layer)].grad is not None and float(edit.u[str(layer)].grad.abs().sum()) > 0


def test_rank1_condition_edits_and_restores(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DIRECTIONS", tmp_path / "directions.safetensors")
    model, _ = tiny
    layer = 1
    u = unit(torch.randn(model.config.hidden_size))
    v = torch.randn(model.config.intermediate_size)
    C.save_directions({f"rank1.u.{layer}": u, f"rank1.v.{layer}": v})
    W = writers(model, layer)["mlp"].weight.detach().clone()
    with ExitStack() as stack:
        C.apply_condition(model, {"edit": {"direction": "rank1", "layers": [layer], "top_k": 3}}, stack)
        got = writers(model, layer)["mlp"].weight.detach()
        changed = (got.float() - W.float()).abs().sum(0) > 1e-5
        assert int(changed.sum()) == 3
        assert torch.allclose(got.float(), rank1_update(W, u, keep_topk(v, 3)).float(), atol=1e-5)
    assert torch.equal(writers(model, layer)["mlp"].weight, W)


def test_manifest_is_the_pilot_set_and_summary_averages_offsets():
    manifest = load_manifest()
    assert manifest["offsets"] == [-2048, -1024, -512, -256, 0, 256]
    assert len(manifest["traces"]) == len({t["id"] for t in manifest["traces"]}) == 32
    rows = [
        {"offset": off, "closed": off > 0, "looping": off < 0}
        for off in manifest["offsets"]
        for _ in range(4)
    ]
    summary = branch_summary(rows, manifest["offsets"])
    assert summary["256"]["closed"] == 1 and summary["-256"]["looping"] == 1 and summary["0"]["n"] == 4

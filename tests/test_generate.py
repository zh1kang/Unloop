import torch

from unloop.generate import Decoding, Sampling, generate_hf, sample

GREEDY = Sampling(temperature=0.0, presence_penalty=0.0, max_new_tokens=12)
PROMPTS = ["5 9 13 7", "3 4", "8 8 21 30 11 6 40"]


def reference(model, prompt, max_new, eos):
    ids = torch.tensor([[int(t) for t in prompt.split()]])
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=0)
    return out[0, ids.shape[1] :].tolist()


def test_batched_greedy_matches_unbatched_hf_generate(tiny):
    model, tok = tiny
    got = generate_hf(model, tok, PROMPTS, GREEDY, batch_size=3)
    for prompt, rec in zip(PROMPTS, got):
        assert rec["tokens"] == reference(model, prompt, 12, eos=1)


def test_rows_that_stop_early_leave_the_others_unchanged(tiny):
    model, tok = tiny
    # make a token that one row emits early act as end-of-sequence, so the batch shrinks mid-generation
    stopper = reference(model, PROMPTS[1], 12, eos=1)[2]
    model.generation_config.eos_token_id = [1, stopper]
    try:
        got = generate_hf(model, tok, PROMPTS, GREEDY, batch_size=3)
    finally:
        model.generation_config.eos_token_id = 1
    finishes = {rec["finish"] for rec in got}
    assert "stop" in finishes
    for prompt, rec in zip(PROMPTS, got):
        assert rec["tokens"] == reference(model, prompt, 12, eos=[1, stopper])


def test_think_budget_forces_the_budget_text(tiny, monkeypatch):
    import unloop.generate as g

    model, tok = tiny
    monkeypatch.setattr(g, "BUDGET_TEXT", "50 51 2")
    got = generate_hf(model, tok, PROMPTS[:1], GREEDY, Decoding(think_budget=4))
    assert got[0]["forced"] and got[0]["tokens"][4:7] == [50, 51, 2]


def test_sample_respects_top_k_top_p_and_penalties():
    gen = torch.Generator().manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]]).repeat(2000, 1)
    seen = torch.zeros_like(logits)
    s = Sampling(temperature=1.0, top_k=3, top_p=0.5, presence_penalty=0.0)
    assert set(sample(logits, seen, s, gen).tolist()) == {0}  # token 0 alone has mass >= 0.5
    s = Sampling(temperature=1.0, top_k=2, top_p=1.0, presence_penalty=0.0)
    assert set(sample(logits, seen, s, gen).tolist()) == {0, 1}
    seen[:, 0] = 1.0  # a penalty of 10 on the seen token 0 makes token 1 the argmax
    s = Sampling(temperature=0.0, presence_penalty=10.0)
    assert set(sample(logits, seen, s, gen).tolist()) == {1}
    bias = torch.ones(2000)
    assert set(sample(logits, seen * 0, s, gen, bias, think_id=3, think_bias=10.0).tolist()) == {3}

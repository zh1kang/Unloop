from unloop.cli import paired_diff, summarize, wilson


def rows(correct, loop, ds="gsm8k", samples=1):
    return [
        {
            "id": f"p{i // samples}",
            "sample": i % samples,
            "dataset": ds,
            "correct": c,
            "loop": lp,
            "think_closed": not lp,
            "finish": "length" if lp else "stop",
            "n_tokens": 10,
        }
        for i, (c, lp) in enumerate(zip(correct, loop))
    ]


def test_summarize_counts_rates():
    m = summarize(rows([1, 0, 0, 1], [0, 1, 1, 0]))["gsm8k"]
    assert m["n"] == 4 and m["correct"] == 0.5 and m["loop"] == 0.5 and m["cut"] == 0.5
    assert m["correct_ci95"][0] < 0.5 < m["correct_ci95"][1]


def test_wilson_bounds():
    assert wilson(0, 10)[0] == 0.0 and 0 < wilson(0, 10)[1] < 0.35
    assert wilson(10, 10)[1] == 1.0


def test_paired_diff_averages_samples_within_problems():
    a = rows([0, 0, 0, 0], [1, 1, 1, 1], samples=2)
    b = rows([1, 1, 1, 0], [0, 0, 0, 1], samples=2)
    mean, lo, hi = paired_diff(a, b, "correct")
    assert mean == 0.75 and lo <= mean <= hi  # problem p0: +1, problem p1: +0.5
    assert paired_diff(a, a, "loop") == [0.0, 0.0, 0.0]

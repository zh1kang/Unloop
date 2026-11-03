import random

from unloop.loops import LoopRule, first_stuck, loop_onset, repeated_mask

RULE = LoopRule(n=4, min_len=40, tau=0.9, stop_window=60, stop_tau=0.98)


def fresh(k, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(10_000) for _ in range(k)]


def test_repeated_mask_marks_second_occurrence():
    assert repeated_mask([1, 2, 3, 1, 2, 3], 3) == [False, False, False, False, False, True]


def test_productive_text_is_not_looping():
    tokens = fresh(500)
    assert loop_onset(tokens, RULE) is None
    assert first_stuck(tokens, RULE) is None


def test_verbatim_cycle_is_looping_with_onset_near_cycle_start():
    prefix, cycle = fresh(200, 1), fresh(13, 2)
    tokens = prefix + cycle * 20
    onset = loop_onset(tokens, RULE)
    assert onset is not None and 200 <= onset <= 200 + 13 + RULE.n
    # the online rule fires one stop window after the cycle repeats
    assert 200 + 13 < first_stuck(tokens, RULE) <= 200 + 13 + RULE.n + RULE.stop_window + 1


def test_near_loop_with_small_variations_counts():
    cycle = fresh(20, 3)
    tokens = fresh(100, 4)
    rng = random.Random(5)
    for _ in range(15):
        tokens += cycle + ([rng.randrange(10_000)] if rng.random() < 0.1 else [])
    assert loop_onset(tokens, RULE) is not None


def test_loop_then_escape_is_not_looping_at_the_end():
    tokens = fresh(100, 6) + fresh(10, 7) * 10 + fresh(300, 8)
    assert loop_onset(tokens, RULE) is None


def test_first_stuck_is_the_first_fully_repeated_window():
    tokens = fresh(150, 9) + fresh(7, 10) * 40
    pos = first_stuck(tokens, RULE)
    mask = repeated_mask(tokens, RULE.n)
    assert pos is not None
    assert sum(mask[pos - RULE.stop_window : pos]) >= RULE.stop_tau * RULE.stop_window
    assert sum(mask[pos - 1 - RULE.stop_window : pos - 1]) < RULE.stop_tau * RULE.stop_window

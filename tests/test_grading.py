import pytest

from unloop.grading import answer_text, extract_answer, is_correct, last_boxed


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so \\boxed{42}.", "42"),
        ("\\boxed{1} then \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("nested \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
        ("\\fbox{7}", "7"),
        ("\\boxed {7}", "7"),
        ("$\\boxed 5$ done", "5"),
        ("cut inside \\boxed{\\frac{1}{", None),
        ("empty \\boxed{}", None),
        ("no box here", None),
    ],
)
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


def test_answer_after_last_think_close():
    response = "draft \\boxed{1}</think>\n\nFinal: \\boxed{2}"
    assert answer_text(response) == "Final: \\boxed{2}"
    assert extract_answer(response) == "2"
    # a closed thinking block without a box after it gives no answer
    assert extract_answer("\\boxed{1}</think>\n\nI am not sure.") is None
    # an unclosed thinking block (cut at the token limit) is read whole
    assert extract_answer("thinking \\boxed{3} wait") == "3"


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("\\frac{1}{2}", "0.5", True),
        ("\\dfrac12", "\\frac{1}{2}", True),
        ("(3, \\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
        ("18", "18", True),
        ("1,000", "1000", True),
        ("x^2+2x+1", "(x+1)^2", True),
        ("17", "18", False),
        (None, "18", False),
        ("", "18", False),
    ],
)
def test_is_correct(pred, gold, ok):
    assert is_correct(pred, gold) is ok

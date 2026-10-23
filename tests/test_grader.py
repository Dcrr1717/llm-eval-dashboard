"""Offline tests for answer extraction and symbolic grading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grader import extract_answer, grade, is_equivalent, normalize

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Some reasoning.\nANSWER: 3/8", "3/8"),
        ("**Final answer:** 42", "42"),
        ("ANSWER: 1\nwait\nANSWER: 2", "2"),  # last explicit answer wins
        ("so the result is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
        ("nested \\boxed{\\sqrt{\\frac{1}{4}}}", "\\sqrt{\\frac{1}{4}}"),
        ("Working...\n\n  23  ", "23"),
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "got, want",
    [
        ("5", "5"),
        ("x+3", "x + 3"),
        ("3 + x", "x + 3"),
        ("3, 2", "2, 3"),
        ("x = 2 or x = 3", "2, 3"),
        ("0.375", "3/8"),
        ("\\frac{\\sqrt{\\pi}}{2}", "sqrt(pi)/2"),
        ("\\dfrac{\\pi^{2}}{6}", "pi^2/6"),
        ("9\\pi", "9*pi"),
        ("$4\\pi/3$", "4*pi/3"),
        ("120°", "120"),
        ("120 degrees", "120"),
        ("1,000", "1000"),
        ("2^(1/2)", "sqrt(2)"),
        ("e^{-1}", "exp(-1)"),
    ],
)
def test_equivalent(got, want):
    assert is_equivalent(got, want)


@pytest.mark.parametrize(
    "got, want",
    [
        ("6", "5"),
        ("x - 3", "x + 3"),
        ("2", "2, 3"),
        ("2, 2", "2, 3"),
        ("pi^2/8", "pi^2/6"),
        (None, "1"),
        ("", "1"),
    ],
)
def test_not_equivalent(got, want):
    assert not is_equivalent(got, want)


@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os').system('echo hi')",
        "().__class__",
        "lambda: 1",
        "open('x')",
        "9^9^9^9",
        "10**100000",
        "1" * 500,
    ],
)
def test_hostile_answers_are_rejected_safely(hostile):
    # Must neither execute anything nor hang; just grade as incorrect.
    assert is_equivalent(hostile, "5") is False


def test_normalize_strips_decoration():
    assert normalize("  $\\left(2\\right)$. ") == "(2)"


def test_bundled_answers_grade_as_correct_against_themselves():
    for line in (ROOT / "problems.jsonl").read_text().splitlines():
        p = json.loads(line)
        extracted, correct = grade(f"Reasoning...\nANSWER: {p['answer']}", p["answer"])
        assert correct, p["id"]
        assert extracted == p["answer"]


def test_sample_extracted_answers_agree_with_correct_flag():
    # The synthetic data is internally consistent: the grader reproduces `correct`.
    for line in (ROOT / "sample_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert is_equivalent(r["extracted"], r["expected"]) == r["correct"], r

"""Offline unit tests for the symbolic grader (no API calls)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grader  # noqa: E402


def test_extract_answer_prefers_marker() -> None:
    text = "Let me work this out.\nSteps...\nANSWER: 42"
    assert grader.extract_answer(text) == "42"


def test_extract_answer_spanish_marker() -> None:
    assert grader.extract_answer("Respuesta: 3/4") == "3/4"


def test_extract_answer_last_marker_wins() -> None:
    text = "ANSWER: 1\nwait, reconsidering\nANSWER: 2"
    assert grader.extract_answer(text) == "2"


def test_extract_answer_fallback_last_line() -> None:
    assert grader.extract_answer("blah blah\n\n  7  ") == "7"


def test_extract_answer_empty() -> None:
    assert grader.extract_answer("") is None
    assert grader.extract_answer("   ") is None


def test_answers_match_numeric_equivalence() -> None:
    assert grader.answers_match("0.5", "1/2")
    assert grader.answers_match("1/2", "0.50")
    assert grader.answers_match("40", "40.0")


def test_answers_match_symbolic_equivalence() -> None:
    assert grader.answers_match("2*x", "2x")
    assert grader.answers_match("2*x + 2", "2*(x+1)")


def test_answers_match_strips_formatting() -> None:
    assert grader.answers_match("55.5", "$55.50")
    assert grader.answers_match("1000", "1,000")
    assert grader.answers_match("4", "x = 4")


def test_answers_do_not_match() -> None:
    assert not grader.answers_match("4", "5")
    assert not grader.answers_match("2*x", "x**2")
    assert not grader.answers_match("4", None)


def test_answers_match_word_fallback() -> None:
    # Non-mathematical answers fall back to a case-insensitive compare.
    assert grader.answers_match("True", "true")
    assert not grader.answers_match("prime", "composite")


def test_grade_end_to_end() -> None:
    correct, extracted = grader.grade("2*x", "The derivative is 2x.\nANSWER: 2x")
    assert correct is True
    assert extracted == "2x"

    wrong, extracted2 = grader.grade("12", "I think it is\nANSWER: 8")
    assert wrong is False
    assert extracted2 == "8"

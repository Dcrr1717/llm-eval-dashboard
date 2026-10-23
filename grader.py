"""Minimal symbolic grader for the optional live "Run eval" tab.

The grader is deliberately small: it extracts a final answer from a model's
free-text response and checks it against a gold answer using SymPy for symbolic
equivalence (so "1/2", "0.5", and "0.50" all match, and "2*x+2" matches
"2*(x+1)"). It is used only by the live tab; the dashboard itself never needs
it and runs fully offline on bundled results.

Nothing in this module calls the Anthropic API or reads any key.
"""

from __future__ import annotations

import re
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Transformations that let "2x" parse as "2*x", etc.
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# Matches an explicit final-answer marker the eval prompt asks the model to use.
# Example model line: "ANSWER: 42" or "Respuesta: 3/4".
_ANSWER_MARKER = re.compile(
    r"(?:answer|respuesta|final)\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_answer(text: str) -> Optional[str]:
    """Extract the model's final answer from its response text.

    Strategy:
        1. Prefer an explicit "ANSWER:"/"Respuesta:" marker (last occurrence).
        2. Otherwise fall back to the last non-empty line of the response.

    Returns:
        The raw answer string, or None if the text is empty.
    """
    if not text or not text.strip():
        return None

    markers = _ANSWER_MARKER.findall(text)
    if markers:
        return markers[-1].strip()

    # Fallback: last non-empty line, stripped of trailing punctuation.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1].rstrip(".")


def _normalise(raw: str) -> str:
    """Clean an answer string for parsing (strip $, commas, trailing period)."""
    cleaned = raw.strip().strip("$").replace(",", "").rstrip(".")
    # Drop a leading "x =" style assignment if present.
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()
    return cleaned


def answers_match(expected: str, extracted: Optional[str]) -> bool:
    """Return True if `extracted` is symbolically equivalent to `expected`.

    Falls back to a case-insensitive string comparison when the values are not
    parseable as mathematical expressions (e.g. word answers).
    """
    if extracted is None:
        return False

    exp_s = _normalise(expected)
    got_s = _normalise(extracted)

    if exp_s == got_s:
        return True

    try:
        exp_expr = parse_expr(exp_s, transformations=_TRANSFORMS)
        got_expr = parse_expr(got_s, transformations=_TRANSFORMS)
        # simplify(a - b) == 0  <=>  symbolic equality.
        return bool(sympy.simplify(exp_expr - got_expr) == 0)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        # Not math expressions: fall back to a lenient string compare.
        return exp_s.casefold() == got_s.casefold()


def grade(expected: str, response_text: str) -> tuple[bool, Optional[str]]:
    """Grade one model response against a gold answer.

    Returns:
        (is_correct, extracted_answer)
    """
    extracted = extract_answer(response_text)
    return answers_match(expected, extracted), extracted

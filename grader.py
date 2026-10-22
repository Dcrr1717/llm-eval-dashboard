"""Minimal symbolic grader: extract a final answer from model text and compare it
to the expected answer with SymPy.

Model output is untrusted, and SymPy's parser evaluates Python, so every answer
is screened (character allowlist, identifier allowlist, length cap) before it
reaches `parse_expr`, and huge exponents are rejected before evaluation.
"""

from __future__ import annotations

import re
from tokenize import TokenError

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

MAX_ANSWER_LEN = 120
MAX_EXPONENT = 1000

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_ALLOWED_CHARS = re.compile(r"^[0-9a-zA-Z+\-*/^().,\s]*$")
_ALLOWED_NAMES = {
    "sqrt", "pi", "e", "E", "exp", "log", "ln", "sin", "cos", "tan", "oo", "inf", "infinity",
    "x", "y", "z", "n", "i", "I",
}
_LOCALS = {"e": sp.E, "E": sp.E, "ln": sp.log, "inf": sp.oo, "infinity": sp.oo, "i": sp.I, "I": sp.I}


class UnsafeAnswer(ValueError):
    """The answer text failed the safety screen and was not parsed."""


def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a model response.

    Priority: an explicit `ANSWER:` line (what the live prompt asks for), then
    the last `\\boxed{...}`, then the last non-empty line.
    """
    if not text:
        return None
    matches = re.findall(r"(?im)^\s*\**\s*(?:final\s+)?answer\s*\**\s*[:=]\s*\**\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, honouring nested braces."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    out = []
    while i < len(text):
        ch = text[i]
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def normalize(answer: str) -> str:
    """Strip LaTeX/markdown decoration and rewrite common notation SymPy can parse."""
    s = answer.strip().strip("*$ ").rstrip(".").strip().strip("*$ ")
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!)", "", s)
    # Rewrite innermost \sqrt{..} / \frac{..}{..} first, repeating to unwrap nesting.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\pi", "pi").replace("π", "pi").replace("\\cdot", "*").replace("×", "*")
    s = s.replace("\\infty", "oo").replace("∞", "oo").replace("−", "-").replace("√", "sqrt")
    s = s.replace("{", "(").replace("}", ")").replace("\\", "")
    s = re.sub(r"°|\bdegrees?\b", "", s)
    # "x = 3" -> "3"; "x = 2 or x = 3" -> "2, 3"
    s = re.sub(r"\b(?:or|and)\b", ",", s)
    s = re.sub(r"\b[a-zA-Z]\s*=\s*", "", s)
    # thousands separators like 1,000 -> 1000 (only when the groups are exactly 3 digits)
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s.replace(" ", "")):
        s = s.replace(",", "")
    return s.strip()


def _safe_parse(expr: str) -> sp.Expr:
    if len(expr) > MAX_ANSWER_LEN:
        raise UnsafeAnswer("answer too long")
    if not _ALLOWED_CHARS.match(expr):
        raise UnsafeAnswer("answer contains disallowed characters")
    for name in re.findall(r"[A-Za-z]+", expr):
        if name not in _ALLOWED_NAMES:
            raise UnsafeAnswer(f"identifier {name!r} not allowed")
    parsed = parse_expr(expr, local_dict=dict(_LOCALS), transformations=_TRANSFORMS, evaluate=False)
    for node in sp.preorder_traversal(parsed):
        if isinstance(node, sp.Pow) and node.exp.is_number:
            if abs(complex(sp.N(node.exp, 15))) > MAX_EXPONENT:
                raise UnsafeAnswer("exponent too large")
    return sp.sympify(parsed)


def _parse_list(answer: str) -> list[sp.Expr]:
    """Parse a comma-separated answer ("2, 3" or "(2, 3)") into expressions."""
    s = answer.strip()
    if s.startswith(("(", "[")) and s.endswith((")", "]")) and "," in s:
        s = s[1:-1]
    return [_safe_parse(part.strip()) for part in s.split(",") if part.strip()]


def _equal(a: sp.Expr, b: sp.Expr, tol: float = 1e-6) -> bool:
    try:
        if sp.simplify(a - b) == 0:
            return True
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        fa, fb = complex(sp.N(a)), complex(sp.N(b))
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def is_equivalent(extracted: str | None, expected: str) -> bool:
    """True when the extracted answer matches the expected one.

    Comma-separated answers are compared as unordered multisets ("3, 2" == "2, 3").
    Falls back to normalized string equality if SymPy can't (or mustn't) parse.
    """
    if extracted is None:
        return False
    got, want = normalize(extracted), normalize(expected)
    if not got:
        return False
    try:
        got_vals, want_vals = _parse_list(got), _parse_list(want)
    except (UnsafeAnswer, SyntaxError, TypeError, ValueError, sp.SympifyError, TokenError):
        return got.replace(" ", "").lower() == want.replace(" ", "").lower()
    if len(got_vals) != len(want_vals):
        return False
    remaining = list(want_vals)
    for g in got_vals:
        match = next((w for w in remaining if _equal(g, w)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def grade(response_text: str, expected: str) -> tuple[str | None, bool]:
    """Extract the answer from a response and grade it. Returns (extracted, correct)."""
    extracted = extract_answer(response_text)
    return extracted, is_equivalent(extracted, expected)


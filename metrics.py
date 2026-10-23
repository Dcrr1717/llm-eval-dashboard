"""Pure data functions for the LLM eval dashboard.

Everything here is Streamlit-free so it can be unit-tested offline:
loading/validating JSONL results, per-model aggregation, the cost-vs-accuracy
Pareto frontier, and the input/output cost split.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import IO

import pandas as pd

# USD per 1M tokens (input, output), Anthropic first-party API list prices.
# Used to split each call's recorded cost into input vs output, and to price
# live runs. Haiku 5.5 is listed at its <=100K-token prompt tier.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Output tokens cost 5x input tokens on every model above; used as a fallback
# split for models missing from PRICES.
DEFAULT_OUTPUT_TO_INPUT_RATIO = 5.0

REQUIRED_COLUMNS = ("id", "model", "correct", "cost")

# Optional columns and the default used when a record omits them.
OPTIONAL_DEFAULTS: dict[str, object] = {
    "topic": "unknown",
    "difficulty": "unknown",
    "tok_in": 0,
    "tok_out": 0,
    "latency_s": float("nan"),
    "expected": None,
    "extracted": None,
}

# Alternative input keys, mapped to the canonical schema above. Includes the
# legacy Spanish keys written by the original eval harness, so its result
# files load unchanged.
ALIASES = {
    "latency": "latency_s",
    "input_tokens": "tok_in",
    "output_tokens": "tok_out",
    "modelo": "model",
    "correcto": "correct",
    "costo": "cost",
    "seg": "latency_s",
    "esperado": "expected",
    "extraido": "extracted",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


class ResultsFormatError(ValueError):
    """Raised when a results file can't be parsed into the expected schema."""


def _to_bool(value: object) -> bool:
    """Coerce the `correct` field, accepting bools, 0/1 and common strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "y", "t"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ResultsFormatError(f"cannot interpret correct={value!r} as a boolean")


def parse_jsonl(lines: Iterable[str]) -> list[dict]:
    """Parse JSONL text lines into dicts, reporting the line number of bad JSON."""
    records: list[dict] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultsFormatError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise ResultsFormatError(f"line {lineno}: expected a JSON object")
        records.append({ALIASES.get(k, k): v for k, v in obj.items()})
    return records


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Validate records and return a typed DataFrame with every expected column."""
    if not records:
        raise ResultsFormatError("no results found (the file is empty)")
    df = pd.DataFrame.from_records(records)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ResultsFormatError(f"missing required field(s): {', '.join(missing)}")
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        elif default is not None:
            df[col] = df[col].where(df[col].notna(), default)

    df["id"] = df["id"].astype(str)
    df["model"] = df["model"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["difficulty"] = df["difficulty"].astype(str).str.lower()
    df["correct"] = df["correct"].map(_to_bool).astype(bool)
    for col in ("cost", "tok_in", "tok_out", "latency_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["cost"].isna().any():
        raise ResultsFormatError("cost must be numeric on every row")
    df["tok_in"] = df["tok_in"].fillna(0).astype(int)
    df["tok_out"] = df["tok_out"].fillna(0).astype(int)
    for col in ("expected", "extracted"):
        # object dtype keeps missing answers as None (pandas' string dtype would turn them into NaN)
        values = [None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in df[col]]
        df[col] = pd.Series(values, index=df.index, dtype=object)
    return df


def load_results(source: str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
    """Load results from a path or a file-like object (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as fh:
            return records_to_frame(parse_jsonl(fh))
    text = source.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return records_to_frame(parse_jsonl(text.splitlines()))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Better behaved than the normal approximation for small n or p near 0/1,
    which is exactly the regime of small eval sets.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model KPIs: accuracy (+95% Wilson CI), total cost, cost/problem, latency.

    Returned sorted by cost per problem (cheapest first).
    """
    rows = []
    for model, g in df.groupby("model", sort=False):
        n = len(g)
        k = int(g["correct"].sum())
        lo, hi = wilson_interval(k, n)
        total = float(g["cost"].sum())
        rows.append(
            {
                "model": model,
                "n": n,
                "correct": k,
                "accuracy": k / n,
                "acc_lo": lo,
                "acc_hi": hi,
                "total_cost": total,
                "cost_per_problem": total / n,
                "mean_latency": float(g["latency_s"].mean()) if g["latency_s"].notna().any() else float("nan"),
                "tok_in": int(g["tok_in"].sum()),
                "tok_out": int(g["tok_out"].sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["cost_per_problem", "model"]).reset_index(drop=True)
    out["pareto"] = pareto_mask(out["cost_per_problem"].tolist(), out["accuracy"].tolist())
    return out


def pareto_mask(costs: list[float], accuracies: list[float]) -> list[bool]:
    """Return True for each point on the cost-vs-accuracy Pareto frontier.

    A point is dominated when another point is at least as cheap AND at least
    as accurate, and strictly better on one of the two. Lower cost and higher
    accuracy are better. Exact duplicates are both kept (neither dominates).
    """
    if len(costs) != len(accuracies):
        raise ValueError("costs and accuracies must have the same length")
    mask = []
    for i, (ci, ai) in enumerate(zip(costs, accuracies)):
        dominated = any(
            cj <= ci and aj >= ai and (cj < ci or aj > ai)
            for j, (cj, aj) in enumerate(zip(costs, accuracies))
            if j != i
        )
        mask.append(not dominated)
    return mask


def accuracy_by(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Accuracy per (model, dimension value), e.g. dimension='difficulty' or 'topic'."""
    out = (
        df.groupby(["model", dimension], sort=False)["correct"]
        .agg(accuracy="mean", n="size", correct="sum")
        .reset_index()
    )
    out["correct"] = out["correct"].astype(int)
    return out


def dimension_order(values: Iterable[str], dimension: str) -> list[str]:
    """Display order for a dimension: easy->medium->hard for difficulty, else alphabetical."""
    uniq = sorted(set(values))
    if dimension != "difficulty":
        return uniq
    known = [d for d in DIFFICULTY_ORDER if d in uniq]
    return known + [d for d in uniq if d not in DIFFICULTY_ORDER]


def split_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Add `cost_in` / `cost_out` columns that always sum to the recorded `cost`.

    The share of each row's cost that came from input vs output tokens is
    taken from the model's list prices (or a 1:5 ratio for unknown models) and
    then scaled to the recorded cost, so caching discounts or price changes in
    the source data never make the breakdown disagree with the totals. Rows
    without token counts are attributed entirely to output.
    """
    out = df.copy()
    pin = out["model"].map(lambda m: PRICES.get(m, (1.0, DEFAULT_OUTPUT_TO_INPUT_RATIO))[0])
    pout = out["model"].map(lambda m: PRICES.get(m, (1.0, DEFAULT_OUTPUT_TO_INPUT_RATIO))[1])
    raw_in = out["tok_in"] * pin
    raw_out = out["tok_out"] * pout
    raw_total = raw_in + raw_out
    share_in = (raw_in / raw_total.where(raw_total > 0)).fillna(0.0)
    out["cost_in"] = out["cost"] * share_in
    out["cost_out"] = out["cost"] - out["cost_in"]
    return out


def cost_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Total input vs output cost per model."""
    split = split_cost(df)
    return split.groupby("model", sort=False)[["cost_in", "cost_out"]].sum().reset_index()


def hardest_problems(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Problems with the lowest accuracy across all models (ties: harder first)."""
    out = (
        df.groupby(["id", "topic", "difficulty"])["correct"]
        .agg(accuracy="mean", models="size")
        .reset_index()
        .sort_values(["accuracy", "id"])
    )
    return out.head(limit).reset_index(drop=True)


def headline(summary: pd.DataFrame) -> str:
    """One-sentence takeaway: which models form the frontier and which are dominated."""
    frontier = summary[summary["pareto"]].sort_values("cost_per_problem")
    dominated = summary[~summary["pareto"]]["model"].tolist()
    chain = " → ".join(frontier["model"])
    text = f"Pareto frontier (cheapest → most accurate): **{chain}**."
    if dominated:
        text += f" Dominated (another model is cheaper *and* at least as accurate): {', '.join(dominated)}."
    return text

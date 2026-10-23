"""Core analytics for the LLM eval dashboard.

This module is deliberately free of any Streamlit / plotting imports so that the
pure data logic (loading, aggregation, Pareto-frontier computation) can be unit
tested offline with no GUI, no network, and no API key.

Data model (one JSON object per line, the format emitted by a sibling eval
harness). Field names follow that harness and are intentionally in Spanish:

    {
      "id": "ALG1",                 # problem id
      "topic": "algebra",           # subject area
      "difficulty": "easy",         # easy | medium | hard
      "modelo": "claude-haiku-5-5", # model under test
      "correcto": true,             # was the extracted answer correct?
      "costo": 0.0004,              # total USD cost of this single call
      "tok_in": 120,                # input (prompt) tokens
      "tok_out": 300,               # output (completion) tokens
      "seg": 1.1,                   # wall-clock latency in seconds
      "esperado": "42",             # OPTIONAL: expected (gold) answer
      "extraido": "42"              # OPTIONAL: answer extracted from the model
    }

`esperado` / `extraido` are optional; the drill-down table shows them when
present and falls back gracefully when they are not.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import IO, Union

import pandas as pd

# Columns the harness is contractually expected to produce.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "topic",
    "difficulty",
    "modelo",
    "correcto",
    "costo",
    "tok_in",
    "tok_out",
    "seg",
)

# Optional columns used by the drill-down view.
OPTIONAL_COLUMNS: tuple[str, ...] = ("esperado", "extraido")

# Published per-1M-token pricing (USD) for the models we expect to evaluate.
# Input price, output price. Used only to split a model's total cost into an
# input-vs-output breakdown; the authoritative per-call total is `costo`.
# Source: Anthropic API pricing (first-party rates).
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


class ResultsFormatError(ValueError):
    """Raised when a results file is missing required columns or is empty."""


def load_results(source: Union[str, bytes, IO[str], IO[bytes]]) -> pd.DataFrame:
    """Load eval results from a JSONL source into a validated DataFrame.

    Args:
        source: A filesystem path, a JSONL string, raw bytes, or any readable
            text/bytes file-like object (e.g. a Streamlit upload buffer).

    Returns:
        A DataFrame with one row per model call, with required columns coerced
        to sensible dtypes (`correcto` -> bool, numeric fields -> float).

    Raises:
        ResultsFormatError: If no valid rows are found or required columns are
            missing.
    """
    text = _read_text(source)
    rows: list[dict] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue  # tolerate blank lines
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultsFormatError(
                f"Invalid JSON on line {line_no}: {exc.msg}"
            ) from exc
        rows.append(obj)

    if not rows:
        raise ResultsFormatError("No result rows found in the provided source.")

    df = pd.DataFrame(rows)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ResultsFormatError(
            f"Results are missing required columns: {', '.join(missing)}"
        )

    return _coerce_dtypes(df)


def _read_text(source: Union[str, bytes, IO[str], IO[bytes]]) -> str:
    """Normalise any accepted source into a JSONL string.

    A short multi-line string is treated as inline JSONL; otherwise a plain
    string is treated as a filesystem path.
    """
    if isinstance(source, bytes):
        return source.decode("utf-8")
    if isinstance(source, str):
        # Heuristic: if it looks like JSONL content, use it directly; else path.
        if "\n" in source or source.lstrip().startswith("{"):
            return source
        with open(source, "r", encoding="utf-8") as fh:
            return fh.read()
    # File-like object (text or binary).
    data = source.read()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns to the dtypes the rest of the module assumes."""
    df = df.copy()
    df["correcto"] = df["correcto"].astype(bool)
    for col in ("costo", "tok_in", "tok_out", "seg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("id", "topic", "difficulty", "modelo"):
        df[col] = df[col].astype(str)
    # Guarantee optional columns exist so downstream code need not branch.
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model KPIs.

    Returns one row per model with:
        - n_problems:      number of calls
        - accuracy:        fraction correct in [0, 1]
        - total_cost:      summed USD cost
        - cost_per_problem: mean USD cost per call
        - mean_latency:    mean wall-clock seconds per call
        - total_tok_in / total_tok_out

    Rows are sorted by accuracy descending for stable, readable display.
    """
    grouped = df.groupby("modelo", sort=False)
    summary = grouped.agg(
        n_problems=("id", "count"),
        accuracy=("correcto", "mean"),
        total_cost=("costo", "sum"),
        cost_per_problem=("costo", "mean"),
        mean_latency=("seg", "mean"),
        total_tok_in=("tok_in", "sum"),
        total_tok_out=("tok_out", "sum"),
    ).reset_index()
    return summary.sort_values("accuracy", ascending=False).reset_index(drop=True)


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag the cost-vs-accuracy Pareto-optimal models.

    We want LOW cost and HIGH accuracy. A model is *dominated* if some other
    model is at least as cheap AND at least as accurate, and strictly better on
    at least one of the two axes. A model is Pareto-optimal when it is not
    dominated by any other.

    The cost axis is `cost_per_problem` (comparable across models regardless of
    how many problems each one ran).

    Args:
        summary: Output of `per_model_summary`.

    Returns:
        A copy of `summary` with an added boolean `pareto` column, sorted by
        cost ascending so the frontier reads left-to-right like the chart.
    """
    out = summary.copy()
    costs = out["cost_per_problem"].to_numpy()
    accs = out["accuracy"].to_numpy()

    pareto_flags: list[bool] = []
    for i in range(len(out)):
        dominated = False
        for j in range(len(out)):
            if i == j:
                continue
            cheaper_or_equal = costs[j] <= costs[i]
            better_or_equal = accs[j] >= accs[i]
            strictly_better = (costs[j] < costs[i]) or (accs[j] > accs[i])
            if cheaper_or_equal and better_or_equal and strictly_better:
                dominated = True
                break
        pareto_flags.append(not dominated)

    out["pareto"] = pareto_flags
    return out.sort_values("cost_per_problem", ascending=True).reset_index(drop=True)


def accuracy_by(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Accuracy per model, broken down by a categorical dimension.

    Args:
        df: Raw results.
        dimension: Column to group by, e.g. "difficulty" or "topic".

    Returns:
        A long-format DataFrame with columns [dimension, "modelo", "accuracy"],
        suitable for a grouped bar chart.
    """
    if dimension not in df.columns:
        raise KeyError(f"Unknown dimension: {dimension!r}")
    grouped = (
        df.groupby([dimension, "modelo"])["correcto"]
        .mean()
        .reset_index()
        .rename(columns={"correcto": "accuracy"})
    )
    return grouped


def cost_breakdown(
    df: pd.DataFrame, pricing: dict[str, dict[str, float]] | None = None
) -> pd.DataFrame:
    """Split each model's spend into input-token cost vs output-token cost.

    The split is derived from token counts and published per-token pricing so
    the stacked bar reflects where the money actually goes. Models with no
    known pricing fall back to apportioning the observed `costo` by token share.

    Args:
        df: Raw results.
        pricing: Optional override of the module-level PRICING table.

    Returns:
        A DataFrame with columns
        [modelo, input_cost, output_cost, total_cost].
    """
    pricing = pricing or PRICING
    grouped = df.groupby("modelo", sort=False).agg(
        tok_in=("tok_in", "sum"),
        tok_out=("tok_out", "sum"),
        observed_cost=("costo", "sum"),
    )

    records: list[dict] = []
    for modelo, row in grouped.iterrows():
        rates = pricing.get(modelo)
        if rates is not None:
            input_cost = row["tok_in"] * rates["input"] / 1_000_000
            output_cost = row["tok_out"] * rates["output"] / 1_000_000
        else:
            # Unknown model: apportion the observed total by token share so the
            # bar still sums to the real spend.
            total_tokens = row["tok_in"] + row["tok_out"]
            if total_tokens > 0:
                input_cost = row["observed_cost"] * row["tok_in"] / total_tokens
                output_cost = row["observed_cost"] * row["tok_out"] / total_tokens
            else:
                input_cost = output_cost = 0.0
        records.append(
            {
                "modelo": modelo,
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "total_cost": float(input_cost + output_cost),
            }
        )

    return pd.DataFrame(records)


def filter_results(
    df: pd.DataFrame,
    modelo: str | None = None,
    topic: str | None = None,
    difficulty: str | None = None,
) -> pd.DataFrame:
    """Filter raw results for the drill-down table.

    Any argument left as None (or the sentinel "Todos"/"All") is not filtered.
    """
    out = df
    for column, value in (
        ("modelo", modelo),
        ("topic", topic),
        ("difficulty", difficulty),
    ):
        if value is not None and value not in ("Todos", "All"):
            out = out[out[column] == value]
    return out.reset_index(drop=True)


def results_to_jsonl(df: pd.DataFrame) -> str:
    """Serialise a results DataFrame back to JSONL (used when appending runs)."""
    buf = StringIO()
    export_cols = list(REQUIRED_COLUMNS) + [
        c for c in OPTIONAL_COLUMNS if c in df.columns
    ]
    for _, row in df[export_cols].iterrows():
        record = {c: row[c] for c in export_cols}
        # Drop NA optional fields for a clean file.
        record = {k: v for k, v in record.items() if not pd.isna(v)}
        buf.write(json.dumps(record, ensure_ascii=False))
        buf.write("\n")
    return buf.getvalue()

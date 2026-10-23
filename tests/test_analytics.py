"""Offline unit tests for the core analytics module.

These tests run with no network, no API key, and no Streamlit. The Pareto
logic is covered with hand-built cases where the correct frontier is obvious.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analytics  # noqa: E402

# A tiny, explicit JSONL fixture: 2 models x 2 problems.
INLINE_JSONL = (
    '{"id":"P1","topic":"algebra","difficulty":"easy","modelo":"m_cheap",'
    '"correcto":true,"costo":0.001,"tok_in":100,"tok_out":200,"seg":1.0,'
    '"esperado":"4","extraido":"4"}\n'
    '{"id":"P2","topic":"algebra","difficulty":"hard","modelo":"m_cheap",'
    '"correcto":false,"costo":0.001,"tok_in":100,"tok_out":200,"seg":1.0,'
    '"esperado":"7","extraido":"5"}\n'
    '{"id":"P1","topic":"algebra","difficulty":"easy","modelo":"m_good",'
    '"correcto":true,"costo":0.010,"tok_in":100,"tok_out":400,"seg":2.0,'
    '"esperado":"4","extraido":"4"}\n'
    '{"id":"P2","topic":"algebra","difficulty":"hard","modelo":"m_good",'
    '"correcto":true,"costo":0.010,"tok_in":100,"tok_out":400,"seg":2.0,'
    '"esperado":"7","extraido":"7"}\n'
)


def test_load_results_parses_and_coerces() -> None:
    df = analytics.load_results(INLINE_JSONL)
    assert len(df) == 4
    assert df["correcto"].dtype == bool
    assert pd.api.types.is_numeric_dtype(df["costo"])
    # Optional columns are guaranteed to exist.
    for col in analytics.OPTIONAL_COLUMNS:
        assert col in df.columns


def test_load_results_tolerates_blank_lines() -> None:
    df = analytics.load_results(INLINE_JSONL + "\n\n")
    assert len(df) == 4


def test_load_results_rejects_missing_columns() -> None:
    bad = '{"id":"P1","modelo":"m","correcto":true}\n'
    with pytest.raises(analytics.ResultsFormatError):
        analytics.load_results(bad)


def test_load_results_rejects_empty() -> None:
    with pytest.raises(analytics.ResultsFormatError):
        analytics.load_results("   \n  \n")


def test_per_model_summary_kpis() -> None:
    df = analytics.load_results(INLINE_JSONL)
    summary = analytics.per_model_summary(df)
    rows = {r["modelo"]: r for _, r in summary.iterrows()}

    assert rows["m_cheap"]["accuracy"] == pytest.approx(0.5)
    assert rows["m_good"]["accuracy"] == pytest.approx(1.0)
    assert rows["m_cheap"]["total_cost"] == pytest.approx(0.002)
    assert rows["m_good"]["cost_per_problem"] == pytest.approx(0.010)
    assert rows["m_good"]["mean_latency"] == pytest.approx(2.0)
    # Sorted by accuracy descending -> m_good first.
    assert summary.iloc[0]["modelo"] == "m_good"


def test_accuracy_by_difficulty() -> None:
    df = analytics.load_results(INLINE_JSONL)
    by_diff = analytics.accuracy_by(df, "difficulty")
    # m_good is correct on both easy and hard.
    good_hard = by_diff[
        (by_diff["modelo"] == "m_good") & (by_diff["difficulty"] == "hard")
    ]["accuracy"].iloc[0]
    cheap_hard = by_diff[
        (by_diff["modelo"] == "m_cheap") & (by_diff["difficulty"] == "hard")
    ]["accuracy"].iloc[0]
    assert good_hard == pytest.approx(1.0)
    assert cheap_hard == pytest.approx(0.0)


def test_accuracy_by_unknown_dimension_raises() -> None:
    df = analytics.load_results(INLINE_JSONL)
    with pytest.raises(KeyError):
        analytics.accuracy_by(df, "nonexistent")


def test_cost_breakdown_sums_to_total_with_known_pricing() -> None:
    df = analytics.load_results(INLINE_JSONL)
    pricing = {
        "m_cheap": {"input": 1.0, "output": 2.0},
        "m_good": {"input": 4.0, "output": 20.0},
    }
    cb = analytics.cost_breakdown(df, pricing=pricing)
    cheap = cb[cb["modelo"] == "m_cheap"].iloc[0]
    # 2 calls * (100 in * $1/1M + 200 out * $2/1M)
    expected_in = 2 * 100 * 1.0 / 1_000_000
    expected_out = 2 * 200 * 2.0 / 1_000_000
    assert cheap["input_cost"] == pytest.approx(expected_in)
    assert cheap["output_cost"] == pytest.approx(expected_out)
    assert cheap["total_cost"] == pytest.approx(expected_in + expected_out)


def test_cost_breakdown_unknown_model_apportions_observed_cost() -> None:
    # No pricing provided for this model -> apportion observed `costo` by tokens.
    cb = analytics.cost_breakdown(
        analytics.load_results(INLINE_JSONL), pricing={}
    )
    cheap = cb[cb["modelo"] == "m_cheap"].iloc[0]
    # observed total for m_cheap = 0.002; tokens 200 in / 400 out -> 1:2 split.
    assert cheap["total_cost"] == pytest.approx(0.002)
    assert cheap["input_cost"] == pytest.approx(0.002 * (200 / 600))
    assert cheap["output_cost"] == pytest.approx(0.002 * (400 / 600))


def test_filter_results() -> None:
    df = analytics.load_results(INLINE_JSONL)
    only_good = analytics.filter_results(df, modelo="m_good")
    assert set(only_good["modelo"]) == {"m_good"}
    # Sentinel values do not filter.
    assert len(analytics.filter_results(df, modelo="Todos")) == 4
    assert len(analytics.filter_results(df, difficulty="hard")) == 2


def test_roundtrip_jsonl() -> None:
    df = analytics.load_results(INLINE_JSONL)
    reparsed = analytics.load_results(analytics.results_to_jsonl(df))
    assert len(reparsed) == len(df)
    assert set(reparsed["modelo"]) == set(df["modelo"])


# --- Pareto frontier: the headline logic, tested on explicit cases. ---------


def _summary(points: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build a minimal summary frame from (model, cost_per_problem, accuracy)."""
    return pd.DataFrame(
        [
            {"modelo": m, "cost_per_problem": c, "accuracy": a}
            for (m, c, a) in points
        ]
    )


def test_pareto_simple_domination() -> None:
    # B is cheaper AND more accurate than A -> A dominated.
    summary = _summary([("A", 0.02, 0.70), ("B", 0.01, 0.90)])
    out = analytics.pareto_frontier(summary)
    flags = dict(zip(out["modelo"], out["pareto"]))
    assert flags["A"] is False
    assert flags["B"] is True


def test_pareto_tradeoff_both_optimal() -> None:
    # Cheap-but-weak vs expensive-but-strong: neither dominates the other.
    summary = _summary([("cheap", 0.001, 0.60), ("strong", 0.015, 0.95)])
    out = analytics.pareto_frontier(summary)
    assert out["pareto"].all()


def test_pareto_equal_accuracy_higher_cost_is_dominated() -> None:
    # Same accuracy, one is pricier -> the pricier one is dominated.
    summary = _summary([("good_cheap", 0.010, 0.88), ("good_pricey", 0.018, 0.88)])
    out = analytics.pareto_frontier(summary)
    flags = dict(zip(out["modelo"], out["pareto"]))
    assert flags["good_cheap"] is True
    assert flags["good_pricey"] is False


def test_pareto_three_tier_frontier() -> None:
    # A classic frontier: each tier buys accuracy with cost; plus one dominated.
    summary = _summary(
        [
            ("haiku", 0.0001, 0.71),
            ("sonnet", 0.0046, 0.81),
            ("opus55", 0.0129, 0.88),
            ("opus48", 0.0183, 0.88),  # same acc as opus55 but pricier
        ]
    )
    out = analytics.pareto_frontier(summary)
    flags = dict(zip(out["modelo"], out["pareto"]))
    assert flags["haiku"] is True
    assert flags["sonnet"] is True
    assert flags["opus55"] is True
    assert flags["opus48"] is False
    # Output is sorted by cost ascending so it reads left-to-right.
    assert list(out["modelo"]) == ["haiku", "sonnet", "opus55", "opus48"]


def test_pareto_single_model_is_optimal() -> None:
    out = analytics.pareto_frontier(_summary([("solo", 0.01, 0.5)]))
    assert bool(out.iloc[0]["pareto"]) is True


def test_pareto_on_bundled_sample() -> None:
    # End-to-end on the shipped sample file: opus-4-8 must be dominated.
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_results.jsonl",
    )
    df = analytics.load_results(sample_path)
    out = analytics.pareto_frontier(analytics.per_model_summary(df))
    flags = dict(zip(out["modelo"], out["pareto"]))
    assert flags["claude-opus-4-8"] is False
    assert flags["claude-haiku-5-5"] is True
    assert flags["claude-opus-5-5"] is True

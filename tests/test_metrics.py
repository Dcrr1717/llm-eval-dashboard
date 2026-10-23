"""Offline tests for data loading, aggregation, the Pareto frontier and cost split."""

from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import pytest

import metrics

ROOT = Path(__file__).resolve().parent.parent


def frame(rows: list[dict]) -> pd.DataFrame:
    return metrics.records_to_frame(rows)


def row(model: str, correct: bool, cost: float, **kw) -> dict:
    return {"id": kw.pop("id", "P1"), "model": model, "correct": correct, "cost": cost, **kw}


# ---------------------------------------------------------------- loading


def test_sample_file_loads_with_expected_schema():
    df = metrics.load_results(ROOT / "sample_results.jsonl")
    assert len(df) == 125
    assert set(df["model"]) == {
        "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5"
    }
    assert df["correct"].dtype == bool
    assert (df["cost"] > 0).all()
    assert set(df["difficulty"]) == {"easy", "medium", "hard"}


def test_load_from_bytes_upload_with_aliases_and_defaults():
    # Second record uses the legacy Spanish keys from the original harness.
    text = (
        '{"id":"A","model":"m1","correct":"true","cost":"0.5"}\n'
        "\n"
        '{"id":"B","modelo":"m1","correcto":0,"costo":0.25,"topic":"algebra","difficulty":"Hard",'
        '"seg":2.5,"esperado":"4","extraido":"5"}\n'
    )
    df = metrics.load_results(io.BytesIO(text.encode()))
    assert list(df["correct"]) == [True, False]
    assert list(df["cost"]) == [0.5, 0.25]
    assert list(df["topic"]) == ["unknown", "algebra"]
    assert list(df["difficulty"]) == ["unknown", "hard"]
    assert list(df["tok_in"]) == [0, 0]
    assert math.isnan(df["latency_s"].iloc[0])
    assert df["latency_s"].iloc[1] == 2.5
    assert list(df["expected"]) == [None, "4"]
    assert list(df["extracted"]) == [None, "5"]


def test_bad_json_reports_line_number():
    with pytest.raises(metrics.ResultsFormatError, match="line 2"):
        metrics.load_results(io.StringIO('{"id":"A","model":"m","correct":true,"cost":1}\n{oops\n'))


def test_missing_required_field_is_reported():
    with pytest.raises(metrics.ResultsFormatError, match="cost"):
        frame([{"id": "A", "model": "m", "correct": True}])


def test_empty_file_is_rejected():
    with pytest.raises(metrics.ResultsFormatError):
        metrics.load_results(io.StringIO("\n\n"))


def test_uninterpretable_boolean_is_rejected():
    with pytest.raises(metrics.ResultsFormatError):
        frame([row("m", "maybe", 0.1)])  # type: ignore[arg-type]


# ---------------------------------------------------------------- pareto


def test_pareto_basic_frontier():
    # (cost, accuracy): A cheap/weak, B mid, C expensive/best, D dominated by B, E dominated by C
    costs = [0.001, 0.01, 0.03, 0.012, 0.05]
    accs = [0.70, 0.85, 0.95, 0.80, 0.90]
    assert metrics.pareto_mask(costs, accs) == [True, True, True, False, False]


def test_pareto_weak_domination_counts():
    # Same cost, lower accuracy -> dominated; same accuracy, higher cost -> dominated.
    assert metrics.pareto_mask([1.0, 1.0], [0.9, 0.8]) == [True, False]
    assert metrics.pareto_mask([1.0, 2.0], [0.9, 0.9]) == [True, False]


def test_pareto_identical_points_both_kept():
    assert metrics.pareto_mask([1.0, 1.0], [0.5, 0.5]) == [True, True]


def test_pareto_single_and_empty():
    assert metrics.pareto_mask([0.3], [0.1]) == [True]
    assert metrics.pareto_mask([], []) == []


def test_pareto_matches_brute_force_definition():
    import random

    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 8)
        costs = [rng.choice([0.1, 0.2, 0.3, 0.4]) for _ in range(n)]
        accs = [rng.choice([0.5, 0.6, 0.7, 0.8]) for _ in range(n)]
        mask = metrics.pareto_mask(costs, accs)
        for i in range(n):
            better_somewhere = any(
                costs[j] <= costs[i] and accs[j] >= accs[i] and (costs[j], accs[j]) != (costs[i], accs[i])
                for j in range(n)
            )
            assert mask[i] == (not better_somewhere)
        # The frontier is never empty and, sorted by cost, strictly increases in accuracy.
        front = sorted({(c, a) for c, a, m in zip(costs, accs, mask) if m})
        assert front
        assert all(a1 < a2 for (_, a1), (_, a2) in zip(front, front[1:]))


def test_pareto_length_mismatch():
    with pytest.raises(ValueError):
        metrics.pareto_mask([1.0], [])


def test_sample_frontier_tells_the_intended_story():
    summary = metrics.model_summary(metrics.load_results(ROOT / "sample_results.jsonl"))
    frontier = summary[summary["pareto"]]["model"].tolist()
    assert frontier == ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


# ---------------------------------------------------------------- aggregation


def test_model_summary_numbers():
    df = frame(
        [
            row("cheap", True, 0.01, latency_s=1.0),
            row("cheap", False, 0.01, latency_s=3.0),
            row("pricey", True, 0.10, latency_s=5.0),
            row("pricey", True, 0.10, latency_s=7.0),
        ]
    )
    s = metrics.model_summary(df).set_index("model")
    assert s.loc["cheap", "accuracy"] == 0.5
    assert s.loc["cheap", "total_cost"] == pytest.approx(0.02)
    assert s.loc["cheap", "cost_per_problem"] == pytest.approx(0.01)
    assert s.loc["cheap", "mean_latency"] == pytest.approx(2.0)
    assert s.loc["pricey", "accuracy"] == 1.0
    assert s.loc["pricey", "n"] == 2
    assert bool(s.loc["cheap", "pareto"]) and bool(s.loc["pricey", "pareto"])
    # sorted cheapest first
    assert metrics.model_summary(df)["model"].tolist() == ["cheap", "pricey"]


def test_model_summary_latency_missing():
    s = metrics.model_summary(frame([row("m", True, 0.1)]))
    assert math.isnan(s.loc[0, "mean_latency"])


def test_wilson_interval_properties():
    lo, hi = metrics.wilson_interval(8, 10)
    assert 0 <= lo < 0.8 < hi <= 1
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = metrics.wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.6
    # Known value: 95% Wilson interval for 8/10 is about (0.490, 0.943).
    lo, hi = metrics.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_accuracy_by_dimension():
    df = frame(
        [
            row("m", True, 0.1, id="a", difficulty="easy"),
            row("m", True, 0.1, id="b", difficulty="easy"),
            row("m", False, 0.1, id="c", difficulty="hard"),
            row("n", True, 0.1, id="c", difficulty="hard"),
        ]
    )
    acc = metrics.accuracy_by(df, "difficulty").set_index(["model", "difficulty"])
    assert acc.loc[("m", "easy"), "accuracy"] == 1.0
    assert acc.loc[("m", "easy"), "n"] == 2
    assert acc.loc[("m", "hard"), "accuracy"] == 0.0
    assert acc.loc[("n", "hard"), "correct"] == 1


def test_dimension_order():
    assert metrics.dimension_order(["hard", "easy", "expert", "medium"], "difficulty") == [
        "easy", "medium", "hard", "expert"
    ]
    assert metrics.dimension_order(["b", "a", "b"], "topic") == ["a", "b"]


def test_split_cost_sums_to_recorded_cost_and_uses_prices():
    df = frame(
        [
            # opus-5-5: $4 in / $20 out per 1M -> 1000 in = 0.004, 1000 out = 0.02
            row("claude-opus-5-5", True, 0.024, tok_in=1000, tok_out=1000),
            # unknown model falls back to a 1:5 price ratio
            row("mystery", True, 0.6, tok_in=100, tok_out=100),
            # no tokens -> whole cost attributed to output
            row("mystery", True, 0.3),
        ]
    )
    split = metrics.split_cost(df)
    assert (split["cost_in"] + split["cost_out"]).tolist() == pytest.approx(df["cost"].tolist())
    assert split.loc[0, "cost_in"] == pytest.approx(0.004)
    assert split.loc[0, "cost_out"] == pytest.approx(0.020)
    assert split.loc[1, "cost_in"] == pytest.approx(0.1)
    assert split.loc[2, "cost_in"] == 0.0


def test_split_cost_scales_to_recorded_cost():
    # Recorded cost is half of list price (e.g. a batch discount): split keeps the ratio.
    df = frame([row("claude-opus-5-5", True, 0.012, tok_in=1000, tok_out=1000)])
    split = metrics.split_cost(df)
    assert split.loc[0, "cost_in"] == pytest.approx(0.002)
    assert split.loc[0, "cost_out"] == pytest.approx(0.010)


def test_cost_breakdown_totals():
    df = metrics.load_results(ROOT / "sample_results.jsonl")
    bd = metrics.cost_breakdown(df).set_index("model")
    totals = df.groupby("model")["cost"].sum()
    for model, total in totals.items():
        assert bd.loc[model, "cost_in"] + bd.loc[model, "cost_out"] == pytest.approx(total)


def test_hardest_problems_sorted():
    df = frame(
        [
            row("m", False, 0.1, id="x"),
            row("n", False, 0.1, id="x"),
            row("m", True, 0.1, id="y"),
            row("n", False, 0.1, id="y"),
        ]
    )
    hard = metrics.hardest_problems(df)
    assert hard["id"].tolist() == ["x", "y"]
    assert hard["accuracy"].tolist() == [0.0, 0.5]


def test_headline_mentions_frontier_and_dominated():
    summary = metrics.model_summary(
        frame([row("a", True, 0.01), row("b", False, 0.02), row("c", True, 0.03)])
    )
    text = metrics.headline(summary)
    assert "a" in text and "Dominated" in text and "b, c" in text

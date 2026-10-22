"""Smoke tests: charts build and the Streamlit app renders offline with the sample data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import charts
import metrics

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sample():
    return metrics.load_results(ROOT / "sample_results.jsonl")


def test_figures_build(sample):
    summary = metrics.model_summary(sample)
    pareto = charts.pareto_chart(summary)
    names = [t.name for t in pareto.data]
    assert names == ["Pareto frontier", "Pareto-optimal", "Dominated"]
    frontier_trace = pareto.data[0]
    assert list(frontier_trace.x) == sorted(frontier_trace.x)  # staircase drawn left to right
    assert len(charts.accuracy_bars(metrics.accuracy_by(sample, "topic"), "topic").data) == 5
    assert len(charts.cost_breakdown_chart(metrics.cost_breakdown(sample)).data) == 2


def test_model_colors_follow_the_model_not_its_rank():
    full = charts.model_colors(["claude-haiku-5-5", "claude-opus-5-5", "zeta"])
    subset = charts.model_colors(["claude-haiku-5-5", "claude-opus-5-5"])
    assert full["claude-opus-5-5"] == subset["claude-opus-5-5"]
    assert full["claude-haiku-5-5"] == subset["claude-haiku-5-5"]
    assert len(set(full.values())) == 3


def test_app_renders_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Pareto frontier" in at.markdown[0].value
    assert len(at.tabs) == 3
    # The live-eval button stays disabled without a key and explicit confirmation.
    run = [b for b in at.button if b.label == "Run eval"][0]
    assert run.disabled
    assert "ANTHROPIC_API_KEY" not in os.environ

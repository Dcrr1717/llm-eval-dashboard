"""Offline tests for the live-eval runner, using a fake client (no network, no key)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import live_eval
import metrics

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = live_eval.load_problems(ROOT / "problems.jsonl")
BY_ID = {p["id"]: p for p in PROBLEMS}
ANSWER_BY_QUESTION = {p["question"]: p["answer"] for p in PROBLEMS}


def fake_response(text: str, stop_reason: str = "end_turn", tok_in: int = 100, tok_out: int = 500):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tok_in, output_tokens=tok_out),
    )


class FakeMessages:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.behaviour(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, behaviour):
        self.messages = FakeMessages(behaviour)


def _api_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(status, request=request), body=None)


def test_problem_bank_is_well_formed():
    assert len(PROBLEMS) == 25
    for p in PROBLEMS:
        assert set(p) == {"id", "topic", "difficulty", "question", "answer"}
        assert p["difficulty"] in {"easy", "medium", "hard"}


def test_solve_one_builds_a_schema_row_and_sends_effort():
    client = FakeClient(lambda kw: fake_response("2x = 22 - 7...\nANSWER: 5", tok_in=120, tok_out=300))
    row = live_eval.solve_one(client, "claude-haiku-5-5", BY_ID["ALG1"], effort="low", max_tokens=2048)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["max_tokens"] == 2048
    assert "temperature" not in sent and "thinking" not in sent

    assert row["correcto"] is True and row["extraido"] == "5"
    # $0.10 / $0.50 per 1M tokens
    assert row["costo"] == pytest.approx(120 * 0.10e-6 + 300 * 0.50e-6)
    # The row is loadable by the dashboard.
    df = metrics.records_to_frame([row])
    assert df.loc[0, "modelo"] == "claude-haiku-5-5"


def test_wrong_answer_and_refusal_are_incorrect():
    client = FakeClient(lambda kw: fake_response("ANSWER: 7"))
    assert live_eval.solve_one(client, "claude-opus-5-5", BY_ID["ALG1"])["correcto"] is False

    client = FakeClient(lambda kw: fake_response("ANSWER: 5", stop_reason="refusal"))
    row = live_eval.solve_one(client, "claude-opus-5-5", BY_ID["ALG1"])
    assert row["correcto"] is False and row["extraido"] is None and row["stop_reason"] == "refusal"


def test_run_eval_covers_all_pairs_and_reports_progress():
    client = FakeClient(lambda kw: fake_response(f"ANSWER: {ANSWER_BY_QUESTION[kw['messages'][0]['content']]}"))
    seen = []
    rows, errors = live_eval.run_eval(
        client, PROBLEMS[:3], ["claude-opus-5-5", "claude-haiku-5-5"], on_progress=lambda d, t, _: seen.append((d, t))
    )
    assert errors == []
    assert len(rows) == 6 and all(r["correcto"] for r in rows)
    assert seen[-1] == (6, 6)



def test_run_eval_continues_on_transient_errors():
    calls = {"n": 0}

    def behaviour(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _api_error(anthropic.RateLimitError, 429)
        return fake_response("ANSWER: 0")

    rows, errors = live_eval.run_eval(FakeClient(behaviour), PROBLEMS[:3], ["claude-opus-5-5"])
    assert len(rows) == 2 and len(errors) == 1 and "rate limited" in errors[0]


def test_run_eval_aborts_on_auth_error():
    client = FakeClient(lambda kw: _api_error(anthropic.AuthenticationError, 401))
    rows, errors = live_eval.run_eval(client, PROBLEMS[:3], ["claude-opus-5-5", "claude-haiku-5-5"])
    assert rows == [] and len(errors) == 1 and "aborted" in errors[0]
    assert len(client.messages.calls) == 1


def test_estimate_cost_bounds():
    typical, worst = live_eval.estimate_cost(["claude-opus-5-5"], n_problems=2, max_tokens=16000)
    assert typical == pytest.approx(2 * (150 * 4e-6 + 1500 * 20e-6))
    assert worst == pytest.approx(2 * (150 * 4e-6 + 16000 * 20e-6))
    assert live_eval.estimate_cost([], 5) == (0, 0)


def test_default_model_and_live_models_are_priced():
    assert live_eval.DEFAULT_MODEL == "claude-opus-5-5"
    assert all(m in metrics.PRICES for m in live_eval.LIVE_MODELS)
